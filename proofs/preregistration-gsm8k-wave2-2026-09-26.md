# Preregistration. GSM8K, wave 2: the 8B and the 14B, Tetra through the served kernel, FP16 and AWQ in vLLM

**Written on 2026-09-26, BEFORE the first job of the wave, and timestamped (`ots stamp`) before
it.** Operator go, 2026-09-26: "bah envoie tout les runs cette nuit comme ça on aura la data",
after the 4B read 82.49 for Tetra, 89.01 for AWQ and 92.12 for FP16, and after the warning that
the 14B would likely not separate its GSM8K gap from its MMLU gap. The cap keeps the rule the
operator chose for wave 1: the sum of the timeouts, **18.90 $**. Project total before: 230.04 $.

Code: commit `53e8167` on branch `gsm8k-raisonnement`. Image: the Space
`Pier-Jean/llvq-runner-cuda` at `97a2b62a`, the image of wave 1. Protocol, grader, dataset
commit, controls: those of `preregistration-gsm8k-campaign-2026-09-26.md` §2 and §4, unchanged,
with the fix of its ECARTS E1 (pyarrow installed in the vLLM job).

## 1. Question

How much GSM8K accuracy do the 8B and 14B sealed files keep through the served kernel, against
FP16 and AWQ w4 g128? Does the gap still exceed the MMLU gap, 5.48 at 8B and 3.22 at 14B
(*measured*, `docs/ETAT.md` §3)?

## 2. The runs

| # | run | what | est. cost | timeout |
|---|---|---|---|---|
| 2 | served 8B | `qwen3-8b-sealed-B.bin` (sha256 `7bdb9a55…`), 1,319 problems | ≈ 2.5 $ | 180 min, 5.40 $ |
| 3 | served 14B | `qwen3-14b-sealed.bin` (sha256 `61db37fe…`), 1,319 problems | ≈ 4.2 $ | 300 min, 9.00 $ |
| 6 | vLLM 8B | FP16 (`Qwen/Qwen3-8B` @ `b968826d`), then AWQ (`Qwen/Qwen3-8B-AWQ` @ `4da05a8e`) | ≈ 0.3 $ | 60 min, 1.80 $ |
| 7 | vLLM 14B | FP16 (`Qwen/Qwen3-14B` @ `40c06982`), then AWQ (`Qwen/Qwen3-14B-AWQ` @ `31c69efc`) | ≈ 0.5 $ | 90 min, 2.70 $ |

Estimates: `ops/gsm8k_plan.py` scaled by what the 4B cost against its estimate (2.13 against
2.52 $ for the kernel, 0.16 against 0.49 $ for vLLM). About 7.5 $ in all.

## 3. The engine gate

Not repeated. Run 4 of wave 1 (FP16 4B, our dense path, against vLLM) decides for the three
sizes, on the assumption that the engine's effect on an FP16 score does not depend on the size.
If it fails, no cross-engine pair of this wave is published either, and an FP16 reference in
our harness at 8B and 14B becomes an operator decision (about 7.6 and 11.9 $, *estimated*).

## 4. Decision rule, per size

G is the paired GSM8K gap, FP16 minus Tetra, with its 95 % interval [lo, hi]. M is the paired
MMLU gap: 5.48 at 8B, 3.22 at 14B.

| outcome | reading |
|---|---|
| lo > M | at this size, reasoning loses more than recall |
| lo ≤ M ≤ hi | no difference between the two losses at this size |
| hi < M | at this size, recall loses more than reasoning |
| otherwise (a run without its trailer, a failed control) | not settled, operator decision |

## 5. Signed predictions

Carried unchanged from `preregistration-gsm8k-pilot-2026-09-26.md` §6 (sha256 `bfd7d75e`),
written before any GSM8K token:

| size | FP16 | AWQ | Tetra | Tetra minus FP16 |
|---|---|---|---|---|
| 8B | 92 [88, 95] | 91 [87, 94] | 82 [75, 88] | −10 [−17, −5] |
| 14B | 94 [91, 96] | 93 [90, 96] | 88 [83, 92] | −6 [−11, −2] |

Updated after the 4B, written now: the 4B gap ran 1.42 times its MMLU gap against FP16 and 1.37
times against AWQ. Applied to the other sizes, Tetra minus FP16 is −7.8 at 8B and −4.6 at 14B,
Tetra minus AWQ −5.8 and −3.4. The 4B reading, "reasoning loses more than recall", should
repeat at 8B and stay unsettled at 14B. Named against me: one ratio from one size, and the row
scales gained half as much at 14B as at 4B and 8B.

## 6. What it will not establish

- Qwen3's reasoning mode, a harder benchmark, a batch above 1, another card.
- The 14B's GSM8K gap against its MMLU gap, if the interval straddles 3.22.
