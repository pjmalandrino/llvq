# Preregistration. GSM8K on the sealed files: the Metal pilot, and the campaign's signed predictions

**Written on 2026-09-26, BEFORE the pilot, and timestamped (`ots stamp`) before its first generated
token.** Operator go, 2026-09-26: "bon prépare le mode avec noyaus full vas y et lance le pilote sur
metal. Ensuite tu me fais un plan de lancement pour les 3 modèles avec temps et couts estimés pour
que je donne un go". The go covers the code and the pilot on the Mac. It covers no paid job.

Cost: $0, on the Mac (Apple M3 Max, 64 GB). Project total before and after: $227.62.
Measured code: base commit `e438ad9` plus the uncommitted files of §8, by sha256.

## 1. Question

How much GSM8K accuracy does each sealed file keep against its FP16 checkpoint, generated through
the served kernel? The pilot does not answer it: 50 problems carry a binomial SE near 6 pp.

The pilot prices the campaign and tests the harness. The campaign's cost rests on three
quantities with no measurement today: the mean length of an answer, the share of answers that
run to the cap, and whether the extraction reads what the models write.

## 2. Setup

Two arms on the Mac, Metal, dense path, f16, batch 1. The same 50 problems for both, drawn by
`gsm8k::select` (seed `0x65736d386b`). Zero-shot, reasoning block pre-filled empty, greedy,
`max_new` 1,024.

```
target/release/oracle Qwen/Qwen3-0.6B 64 metal
shasum -a 256 ~/q4b-sealed-2026-09-23/qwen3-4b-sealed.bin
export LLVQ_DATASET_REV=740312add88f781978c0658806c59bc2815b9866
nice -n 10 env LLVQ_GSM8K_DUMP=docs/data/gsm8k-dumps/pilot-4b-f16-metal.jsonl \
  target/release/gsm8k Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c metal 50
nice -n 10 env LLVQ_GSM8K_DUMP=docs/data/gsm8k-dumps/pilot-4b-sealed-metal.jsonl \
  target/release/gsm8k ~/q4b-sealed-2026-09-23/qwen3-4b-sealed.bin metal 50
target/release/gsm8kpair docs/data/gsm8k-dumps/pilot-4b-f16-metal.jsonl \
  docs/data/gsm8k-dumps/pilot-4b-sealed-metal.jsonl
uv run ops/gsm8k_vllm.py --arm f16 --model Qwen/Qwen3-4B \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c --limit 50 \
  --dataset-rev 740312add88f781978c0658806c59bc2815b9866 \
  --check docs/data/gsm8k-dumps/pilot-4b-f16-metal.jsonl --check-only
```

`target/release/*` built from the code of §8 with `--features metal`: `gsm8k` sha256 starts
`922aa36875cd3ac4`, `gsm8kpair` `5378e589ccc3109a`. `openai/gsm8k` is read at commit
`740312ad`, the revision `main` resolved to on 2026-09-26.

Arm A is the FP16 checkpoint at the revision cached on the Mac. Arm B is the paper-2 4B object,
`qwen3-4b-sealed.bin`, read as its dense reconstruction. No sealed file passes the Metal fused
path, which walls on the int4 `down_proj` at `d_in` 8,192.

## 3. Controls

1. `oracle` MATCH on Metal before the first problem (hard rule 10).
2. The sealed file's sha256 starts `886391a8c03f66dc` (`configs/README.md`).
3. Both dumps hold the same 50 problems, the same `qhash` problem by problem, the same run
   fingerprint, and their trailer. `gsm8kpair` refuses the pair otherwise.
4. `ops/gsm8k_vllm.py --check-only` rebuilds the same 50 prompts. That ties the campaign's vLLM
   arms to these token ids before any paid job. Run before this stamp without a dump, it drew
   the sample and printed the run fingerprint `94c7b98876288901`. Both Rust dumps must end on
   that fingerprint.
5. Every row not read from a box, every row stopped at the cap, and every FP16 row graded wrong
   is read by hand and classed as a model error or an extraction error. The counts go in the
   journal.
6. The grading module passed a mutation pass before this stamp, 12 mutants of 12 killed (§8).

## 4. What gets published, and what does not get compared

Published in the journal only: per arm, the generated tokens (mean, median, p90, max), the cap
stops, the extraction sources, the hand-audit counts, the Metal timings.

Not a result: the two accuracies and their difference. At n = 50 the SE is near 6 pp, and the
pilot enters no table of the paper or of `docs/ETAT.md`.

Not transferred: the Metal milliseconds. The campaign is priced on the L40S journals
(`paper-table-2026-09-25`, `f1e-census-2026-09-11`) with the lengths measured here.

## 5. Decision rule

The first row that matches applies.

| outcome | reading |
|---|---|
| the hand audit finds an extraction error | the rule is fixed in `gsm8k.rs` and both dumps are re-graded from their completions, without a rerun; the deviation is written; the table is read again |
| FP16 below 70 % after any re-grade | the protocol is suspect (template, stop, instruction); no campaign is priced before a diagnosis |
| the sealed arm stops at the cap on more than 5 of 50 | the plan prices that share at the cap, and `max_new` becomes an operator decision |
| otherwise | the plan is priced on each arm's measured mean length |

## 6. Signed predictions

Pilot, 50 problems:

| arm | accuracy | mean generated tokens | cap stops |
|---|---|---|---|
| FP16 4B | 90 % [80, 96] | 260 [150, 420] | 0 [0, 1] |
| sealed 4B, dense | 74 % [58, 86] | 290 [160, 500] | 1 [0, 4] |

Campaign, full test split, our files through the served kernel. Written now, before any GSM8K
token, so the pilot cannot move them:

| size | FP16 | AWQ, vLLM | ours, kernel | ours minus FP16 |
|---|---|---|---|---|
| 4B | 90 [86, 94] | 88 [84, 92] | 75 [67, 83] | −15 [−23, −8] |
| 8B | 92 [88, 95] | 91 [87, 94] | 82 [75, 88] | −10 [−17, −5] |
| 14B | 94 [91, 96] | 93 [90, 96] | 88 [83, 92] | −6 [−11, −2] |

Reasoning. Our MMLU gaps to FP16 are 6.77, 5.48 and 3.22 pp. MMLU reads one logit per question;
a GSM8K answer chains five to eight steps, and an error at one step carries to the next. The
August per-subject profile already put abstract algebra at chance. I expect a GSM8K gap near
twice the MMLU gap, shrinking with size as the MMLU gap does.

Named against me: no 2-bit Qwen3 has been scored on GSM8K in this repository, and the FP16
levels above are recollections of published figures, not measurements. A GSM8K gap below the
MMLU gap at any size would refute the compounding argument.

## 7. What it will not establish

- Anything about the served kernel: no sealed file runs fused on Metal.
- Any accuracy: 50 problems.
- Any time on the L40S.

## 8. Code at stamp time

Base commit `e438ad9`. Uncommitted at stamp time, by sha256:

| file | sha256 |
|---|---|
| `llvq-llm/src/chatfmt.rs` (new, moved out of `bin/chat.rs`) | `8a88e51357d4701d` |
| `llvq-llm/src/gsm8k.rs` (new) | `3aa12d33f688518d` |
| `llvq-llm/src/bin/gsm8k.rs` (new) | `4d916f06535c54b4` |
| `llvq-llm/src/bin/gsm8kpair.rs` (new) | `0fcff357863b3579` |
| `llvq-llm/src/bin/chat.rs` | `1d115adc784c0311` |
| `llvq-llm/src/lib.rs` | `959beb8f36ae376d` |
| `llvq-llm/src/corpus.rs` | `0f101cbe592dc32c` |
| `ops/gsm8k_vllm.py` (new) | `6196fd2bdce2f0af` |
| `ops/Dockerfile.cuda` | `adab7feaabe8cbab` |

Tests: 17 in `gsm8k`, 3 in `bin/gsm8k`, 8 in `bin/chat`, all passing. `cargo clippy -p
llvq-llm --all-targets --features metal` gives zero warnings.

Mutation pass on `gsm8k.rs`, restored from a byte copy after each mutant and checked by digest.
Ten mutants at digest `04410b693367`: first box instead of last, trailing zeros kept, first
number in the box, any answer right, `qhash` not checked, sample not sorted, trailer optional,
first `####` instead of last, unclosed box accepted, fraction numerator read. Two more at the
stamped digest: re-grade keeps the raw gold, pairing ignores the gold. All twelve killed by a
failing test, none by a compile error.
