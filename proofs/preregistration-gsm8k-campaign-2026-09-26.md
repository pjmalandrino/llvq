# Preregistration. GSM8K, wave 1: the smoke through the served kernel, then the 4B with Tetra, AWQ and FP16

**Written on 2026-09-26, BEFORE the first job, and timestamped (`ots stamp`) before it.**
Operator go, 2026-09-26: "ok pour le smoke mais en suivant si ok je vuex le 4B avec Tetra AWQ
et f16; après on verra". Asked next, the operator answered: the engine gate included ("Oui,
inclus"), and the cap of this wave is the sum of its timeouts ("La somme des timeouts").

Announced cost: smoke 0.33 $, then 8.30 $ for runs 1, 4 and 5, so 8.63 $ in all (*computed*,
[gsm8k-pilot](../docs/mesures/gsm8k-pilot-2026-09-26.txt), `plan.txt`). Cap in force for this
wave: **15.75 $**, the sum of the four timeouts. Project total before: 227.62 $.
Code: commit `685b32a` on branch `gsm8k-raisonnement`. Image: the Space
`Pier-Jean/llvq-runner-cuda` rebuilt from the commit that carries this file; its sha goes in the
journal.

## 1. Question

How much GSM8K accuracy does the 4B sealed file keep through the served kernel, against FP16
and AWQ w4 g128? Is the loss larger than its MMLU loss, 6.77 pp below FP16 and 4.76 below AWQ
(*measured*, `docs/ETAT.md` §3)?

## 2. The runs

| # | run | image | what | est. billed | est. cost | timeout |
|---|---|---|---|---|---|---|
| 0 | smoke | `llvq-runner-cuda` | `qwen3-4b-sealed.bin` through the served kernel, the pilot's 50 problems | 11 min | 0.33 $ | 30 min, 0.90 $ |
| 1 | served 4B | same | the same file, 1,319 problems | 84 min | 2.52 $ | 150 min, 4.50 $ |
| 4 | engine gate | same | `Qwen/Qwen3-4B` @ `1cfa9a72` through our dense path, 1,319 problems | 176 min | 5.29 $ | 300 min, 9.00 $ |
| 5 | vLLM 4B | `vllm/vllm-openai:v0.26.0` @ `sha256:ffb2d59b` | FP16 (`Qwen/Qwen3-4B` @ `1cfa9a72`), then AWQ (`Qwen/Qwen3-4B-AWQ` @ `74d4bd2b`, `awq_marlin`), 1,319 problems each | 16 min | 0.49 $ | 45 min, 1.35 $ |

The numbers keep the plan's: runs 2, 3, 6 and 7 are the 8B and 14B, not in this wave.

```
SIZE=4b LIMIT=50 IMAGE_SHA=<sha> bash ops/jobs/gsm8k-served.sh            # run 0
SIZE=4b IMAGE_SHA=<sha> bash ops/jobs/gsm8k-served.sh                     # run 1
SIZE=4b ARM=f16-dense IMAGE_SHA=<sha> bash ops/jobs/gsm8k-served.sh       # run 4
SIZE=4b bash ops/jobs/gsm8k-vllm.sh upload && SIZE=4b bash ops/jobs/gsm8k-vllm.sh   # run 5
```

Every arm: zero-shot, Qwen3 chat template, reasoning block pre-filled empty, greedy, stop on
`<|im_end|>` or `<|endoftext|>`, `max_new` 1,024, `openai/gsm8k` test at commit `740312ad`.
Prompt ids come from `llvq_llm::chatfmt` or from `ops/gsm8k_vllm.py` (sha256 `907ee931`), and
`qhash` proves them equal problem by problem. One grader, `gsm8kpair`, re-grades every row.

## 3. The smoke decides whether runs 1, 4 and 5 start

Run 0 passes when all five hold:

1. `oracle` MATCH on the card, the object's sha256 `886391a8…`, the config's sha256 checked.
2. The dump carries its trailer and the pilot's run fingerprint, `94c7b98876288901`.
3. At least 30 of 50 right. The dense reconstruction read 40 on Metal, and a broken kernel
   path reads near zero.
4. At most 5 of 50 stopped at the cap.
5. The 4B census projected from the smoke's own prefill and decode milliseconds, with the
   pilot's lengths, fits 120 min, 80 % of run 1's timeout.

All five pass: runs 1, 4 and 5 start together, with no second go. One fails: nothing starts,
and the operator decides.

Also read from the smoke, descriptive only: the served decode at GSM8K lengths, and the smoke
paired with `pilot-4b-sealed-metal.jsonl`, the same file through two paths on two backends.

## 4. Controls of runs 1, 4 and 5

1. `oracle` MATCH on runs 1 and 4 (hard rule 10).
2. Every dump carries its trailer and the census fingerprint `bfa9135ce4b55c75`, the one the
   Python builder prints with all six reference tokenizers.
3. `gsm8kpair` re-grades 0 rows of a Rust dump and every row of a vLLM dump.
4. The vLLM log names Marlin on the AWQ arm and float16 on both arms.

If one fails, no number of the failing run is published.

## 5. The engine gate: run 4 against the FP16 arm of run 5

| outcome | reading |
|---|---|
| \|Δ\| ≤ 1.5 pp and McNemar p ≥ 0.05 | the cross-engine pairs, our kernel against vLLM, are published and labelled "engines differ" |
| otherwise | no cross-engine pair is published; Tetra is compared to run 4's FP16 only, and an AWQ arm on our dense path becomes an operator decision, about 5.3 $ |

Run 4 is a same-harness FP16 reference whatever the gate reads.

## 6. Decision rule for the 4B

G is the paired GSM8K gap, FP16 minus Tetra, with its 95 % interval [lo, hi]: against run 5's
FP16 if the gate passes, against run 4's otherwise. M = 6.77 pp, the paired MMLU gap.

| outcome | reading |
|---|---|
| lo > M | at 4B, reasoning loses more than recall; the paper and the card say so |
| lo ≤ M ≤ hi | no difference between the two losses at 4B |
| hi < M | at 4B, recall loses more than reasoning; the August profile does not carry to the sealed file |
| otherwise (a run without its trailer, a failed control) | not settled, operator decision |

## 7. Signed predictions

For the 4B, carried unchanged from `preregistration-gsm8k-pilot-2026-09-26.md` §6 (sha256
`bfd7d75e`, stamped before any GSM8K token): FP16 90 [86, 94], AWQ 88 [84, 92], Tetra 75
[67, 83], Tetra minus FP16 −15 [−23, −8].

For the smoke, new, written after the pilot: 40 of 50 right [32, 46], mean length 320 tokens
[280, 370], served decode 9.5 ms a token [8.8, 11.0] at these lengths. Reasoning: the kernel
matched the dense reconstruction on MMLU (3 discordant of 2,280), and past 256 tokens the
attention and the growing cache add a few percent a token. Named against me: two backends
decode different chains, so 50 problems can move by several either way.

## 8. What it will not establish

- The 8B and the 14B: a later wave, an operator decision.
- Qwen3's reasoning mode (`<think>`).
- A batch above 1 on our side, a context past 1,400 tokens, another card than the L40S.
