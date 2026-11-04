DRAFT, not stamped, not committed.

# Preregistration. G0: what the published drafters accept against the sealed 4B, measured on the Mac

**Written on 2026-11-02 as a draft, revised the same day after an adversarial review. To be committed and
TIMESTAMPED (`ots stamp`) BEFORE the first run.** No operator go yet.
- Cost: **$0**, on the operator's M3 Max (64 GB, 16 cores).
- Wall clock: about 10-13 h over one or two nights at `nice -n 10` and 12 torch threads, plus a 1-2 h pilot whose
  references the full run reuses (*estimated*, §8).
- Spent before it: $248.84 over 237 lines of `docs/data/jobs.csv` (*measured*).
- Ruling R1 comes first: the operator authorizes this measurement of the speculative lead closed on 2026-08-05
  (`docs/archive/pistes-facteurs-cles-2026-08-05.md:75`), or lifts that verdict.

## 1. Why this run exists

Every speculative ceiling in the wave-2 program multiplies a τ taken from DeepSpec's Table 1, measured against BF16
Qwen3-4B at T = 1. Nothing says what the same drafters accept against the sealed 4B at about 2.7 b/param, greedy. The
only related number is the bare artifact's teacher-forced top-1 agreement with dense: 74.45 % on WikiText-2
(*measured*, `kl-temperature-4b-2026-09-16.txt`), on an object without row scales or int4 parts.

Three program decisions wait on it:
- whether neural drafting is worth an engine integration;
- whether a fine-tune is owed;
- whether the drafter can share our int4 head (rule 6).

**This run carries a measurement, not a gate** (METHODE §1). τ is a proxy for throughput. Its decision rules route
the next work; they adopt nothing and kill nothing. The fundamental gate, end-to-end tok/s with identical output, sits
in the engine job E0 (`prereg-speculation-2.md`) and in the in-engine drafter job.

## 2. What the code holds

`code/speculation/g0_tau.py` (to be committed under `ops/` with this file) is a PEP-723 script run with `uv run`.

**Vendored DeepSpec**, at commit `005e03b81cec38b7da6399833d609ee89a2587f2` (MIT):
- code: `utils/sampling.py`, `data/parser.py`, `modeling/dspark/{common,markov_head}.py`,
  `modeling/dspark/qwen3/{config,modeling}.py`, `modeling/eagle3/{common.py,qwen3/modeling.py}`,
  `eval/base_evaluator.py`, `eval/dspark/draft_ops.py`;
- data: `eval_datasets/mt-bench.jsonl`, `eval_datasets/humaneval.jsonl`, `eval_datasets/gsm8k.jsonl`. These are
  Table 1's inputs: first turn only, GSM8K shuffled by `random.Random(980406)` and cut to 500
  (`base_evaluator.py:31-55`, `:513-525`).

The sha256 of each file is printed at the first fetch and copied into the table below before stamping:

| file | sha256 |
|---|---|
| each of the 11 code files and 3 dataset files | (filled at L1, before stamping) |
| `corpora.jsonl` | (filled before stamping) |

`corpora.jsonl` holds one line per prompt: id, corpus, messages, enable_thinking, source and license.
- Chat (80), code (164) and math (80) are rows of the vendored datasets, named by dataset and row index.
- Extraction: 40 English and 40 French prompts on openly licensed documents of 1-4k tokens, 10 of them at 6-8k.
- The product slice (≥ 10 operator documents, ruling R2) enters as per-document sha256 only. The text stays local
  and outside every committed file. If R2 says no, 10 synthetic French documents replace it, pinned like the rest.

Package `__init__` files and `deepspec.utils` are replaced by stubs that export only what those files import. Any shim
of a `transformers` name is printed and listed in the journal. `transformers` is pinned at 5.18.0: its `Cache.crop`
reads a positive argument as an absolute size only on a deprecated path. So every cache the vendored code creates is
a subclass whose `crop(n)` keeps n positions, injected by name and listed as a shim.

Pinned objects:
- Sealed 4B: `Pier-Jean/Qwen3-4B-LLVQ-Tetra` at `acc5f663ea645886febe41c8abb39fea2b78c4a6`, read by llvq-tetra 0.2.0
  without `device_map` (dense rebuild), cast to f16, moved to `mps`.
- Dense 4B: `Qwen/Qwen3-4B` at `1cfa9a7208912126459214e8b04321603b3df60c`, loaded on the CPU, f16, moved to `mps`.
  Its BF16 safetensors are also read in their own dtype for V4.
- Drafters: `deepseek-ai/dspark_qwen3_4b_block7` at `3457dff`, `dflash_qwen3_4b_block7` at `02d530b`,
  `eagle3_qwen3_4b_ttt7` at `b0b90fd`, bf16 as published.
- Optional (stage 4c): the served kernel's GSM8K dump and the FP16 dense arm's
  (`docs/mesures/gsm8k-wave1-2026-09-26-brut/run1-served/gsm8k-4b-served.jsonl` and
  `run4-f16dense/gsm8k-4b-f16-dense.jsonl`), each with its sha256 copied here before stamping.

**Head arms.**
- `own`: the drafter's published BF16 frozen copy.
- `shared`: the sealed table in `lm_head` and `embed_tokens`, mask row 151669 included.
- `shared-bf16mask`: the sealed table, with row 151669 kept from the drafter's own BF16 table (5,120 bytes).

In both shared arms, logits are f32 products against the f16 table, never a bf16 copy of it. The primary arm is
`shared-bf16mask`, the cheapest faithful deployment: the head is ours, the mask input is the one the drafter was
trained on.

## 3. What runs

Stages in order, one process each. Every arm of stages 6-9 runs the same prompts (80 + 164 + 80 + 80 + 10), and every
arm reads the same reference continuation, so arms are paired by prompt. Greedy everywhere except V1 and the thinking
slice. Prompt ids are split by parity: every choice made on data (gate, k, drafter, confidence temperature) is made
on even ids and scored on odd ones.

1. **L1 checks** (voiding, §4).
   - V3: taps from `output_hidden_states` equal forward hooks on decoder layers 1, 9, 17, 25 and 33, bitwise in f32.
   - V4: each drafter's `lm_head` and `embed_tokens` equal Qwen3-4B's BF16 checkpoint table, read in its own dtype,
     max |Δ| = 0.
   - Printed, not voiding: max |Δ| and cosine of row 151669, sealed against BF16.
2. **Pilot, an interim look declared here.** DSpark, `shared-bf16mask`, both targets, MT-Bench ids 0-39 and
   HumanEval ids 0-39, stages 4, 5 and 6 restricted to those prompts. Rule D0 reads it. Its references and taps are
   reused by the full run.
3. **V1, the harness against the authors.** BF16 target in bf16, T = 1, the vendored online loop
   (`generate_decoding_sample`) with the DSpark evaluator's three callbacks, seed `980406 + idx`, max 2,048 new
   tokens. Sets: MT-Bench 80, HumanEval 164, the first 100 of DeepSpec's seeded GSM8K 500, all from the vendored
   files.
4. **References.** Each target's plain greedy continuation: single-row decode with a cache, max 512 new tokens,
   stop on 151643 or 151645, `enable_thinking=False`. A prompt is flagged as looping when a 16-gram occurs 3 times;
   the onset is the third occurrence.
   - 4b. **Prompt lookup, right after the references**, minutes on token lists: c and τ_d per gate m ∈ {2, 3, 4} and
     k ∈ {2, 3, 4, 8}, on loop-truncated text, both targets. D3 is routed here with L1a's β
     (`prereg-speculation-0.md`) if that run exists.
   - 4c. **Optional, $0: the served object's own GSM8K text.**
     - Round trip. Rebuild each prompt with `bin/gsm8k`'s template (`chatfmt.rs`, `gsm8k::user_body`, thinking off)
       and require len = `n_prompt` and the `qhash` fingerprint, through a port of `eval::token_fingerprint` checked
       on three rows first. Re-encode each completion and require len = `n_gen` and an exact decode. Failures are
       excluded and counted.
     - Agreement. Teacher-force the served completions through the sealed rebuild: top-1 agreement and the
       first-divergence position per problem, read by D7.
     - Optionally, FP16 completions through the dense target. The first 160 problems by index of each then serve as
       math references from the real objects.
5. **Taps.** One teacher-forced forward per (prompt, target) over prompt plus reference,
   `output_hidden_states=True`.
6. **Offline replay** (`speculation.md` §3.1).
   - Sealed target × {DSpark, DFlash} × {own, shared, shared-bf16mask}.
   - Sealed target × EAGLE-3 × {own, shared}. EAGLE-3 drafts autoregressively and reads no mask row, so its
     `shared-bf16mask` arm would equal `shared`.
   - Dense target × the three drafters, own head.
   - If 4c ran: sealed × DSpark × shared-bf16mask on the served references, and dense × DSpark × own on the FP16
     references.
7. **V2.** Online T = 0 loop against the offline replay, DSpark, both targets, 20 prompts.
8. **Agreement.** Teacher-forced top-1 and full-vocabulary KL(dense ‖ sealed), log-softmax in f32, on both targets'
   continuations and on 12 WikiText-2 test windows of 4,096 tokens (fingerprint printed). Tap cosine per layer.
9. **Probes.** Sealed target, `shared-bf16mask`, chat, code and extraction. Two DSpark bodies: RTN int8 per channel,
   and RTN int4 g128 (4.25 b/weight). And a drafter context limited to the last 2,048 positions on the 6-8k slice.
   No probe at Tetra's 2.15 b/weight.
10. **Thinking slice**, informational: 20 GSM8K and 20 MT-Bench prompts, `enable_thinking=True`, T = 0.6 without
    top-p or top-k, online loop, DSpark, `shared-bf16mask`, max 2,048 new tokens.

## 4. Gates, each one voids the run

1. The sealed model's `quantization_config.artifact_sha256` starts with `886391a8`, and llvq-tetra prints 252 records
   rebuilt with no missing or unexpected key.
2. V3 and V4 pass bitwise.
3. Every vendored file (code and datasets), `corpora.jsonl` and any optional dump matches the sha256 copied into §2;
   every shim is listed.
4. V1: on at least 2 of 3 sets, |τ − τ_Table1| ≤ max(0.15, 2.5 × bootstrap SE over prompts), Table 1 being DSpark
   3.64 (MT-Bench), 5.38 (HumanEval), 6.11 (GSM8K). Otherwise the harness is wrong: fix it, write the deviation,
   rerun.
5. V2: offline and online τ within ±3 % on each target, and the same accepted count on ≥ 95 % of rounds up to the
   first text divergence.

Machine load voids nothing: no output of this run depends on timing. A `ps` snapshot is logged each hour as
information, and every wall-clock figure in the journal is informational.

## 5. Outcomes reported, never gated

- Per (target, drafter, head arm, corpus):
  - τ, DeepSpec's metric: mean accepted + 1, accepted only on a stop round;
  - the per-position acceptance curve, rounds and tokens;
  - τ on loop-truncated text.
- Prompt lookup: c and τ_d per gate and k, on loop-truncated text.
- S over the β grid {0.03, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80}, for fixed k, the best k (chosen on even
  ids) and confidence-scheduled K. Also S at L1a's β if it exists.
- Agreement and KL per corpus; per-layer tap cosine; probe Δτ; the mask-row print.
- Thinking-slice τ and loop rate; 4c's round-trip counts, agreement and first-divergence quantiles.

## 6. Signed predictions

| quantity | point | interval | label |
|---|---|---|---|
| V1 DSpark τ, BF16, T = 1, MT-Bench / HumanEval / GSM8K-100 | 3.6 / 5.4 / 6.1 | [3.40, 3.85] / [5.10, 5.65] / [5.75, 6.40] | *estimated* from Table 1 |
| τ_dense(T = 0) / τ_dense(T = 1), DSpark, chat | 1.07 | [1.00, 1.15] | *estimated*, DFlash-b16 4.35/4.07 |
| pilot q_τ, DSpark, chat / code | 0.80 / 0.85 | [0.65, 0.95] / [0.70, 0.98] | *estimated* |
| **q_τ, best drafter, shared-bf16mask, chat (odd ids)** | **0.80** | [0.65, 0.95] | *estimated* |
| **q_τ, best drafter, shared-bf16mask, code (odd ids)** | **0.85** | [0.70, 0.98] | *estimated* |
| q_τ, same, math | 0.88 | [0.75, 1.00] | *estimated* |
| Δτ = τ(shared-bf16mask) − τ(own), sealed, each corpus | 0.00 | [−0.10, +0.15] | *estimated* |
| Δτ = τ(shared) − τ(shared-bf16mask), sealed, each corpus | 0.00 | [−0.15, +0.05] | *estimated* |
| cosine of row 151669, sealed against BF16 | 0.95 | [0.80, 0.995] | *estimated* |
| order of τ, sealed, every corpus | DSpark > DFlash > EAGLE-3 | | *estimated*, Table 1 order |
| top-1 agreement, sealed continuations, chat | 0.88 | [0.80, 0.94] | *estimated* |
| top-1 agreement, WikiText-2 | 0.78 | [0.75, 0.83] | *estimated*; above the bare artifact's 0.7445 |
| KL(dense ‖ sealed), WikiText-2, nats a position | 0.50 | [0.35, 0.62] | *estimated*; below the bare artifact's 0.628 |
| loop rate, sealed, chat / code / extraction | 3 % / 2 % / 5 % | [0, 8] / [0, 6] / [0, 12] % | *estimated* |
| prompt lookup, extraction, sealed, gate 3: c / τ_d(k = 4) | 0.45 / 2.6 | [0.25, 0.65] / [2.0, 3.3] | *estimated* |
| prompt lookup, chat: c | 0.15 | [0.05, 0.30] | *estimated* |
| S of prompt lookup on extraction at β = 0.58, best k | 1.05 | [0.96, 1.20] | *estimated* |
| probe Δτ/τ, int8 / int4 g128 | −1 % / −8 % | [−4, +1] / [−20, −2] % | *estimated* |
| probe Δτ/τ, context 2,048 on the 6-8k slice | −5 % | [−20, 0] % | *estimated* |
| 4c: round trip passes | 98 % of rows | [90, 100] % | *estimated* |
| 4c: top-1 agreement, sealed rebuild against served text | 0.99 | [0.97, 0.997] | *estimated*; 3 discordant MMLU answers in 2,280 between dense arm and kernel (*measured*, census 2026-09-11) |
| 4c: median first-divergence position, tokens | 80 | [30, 250] | *estimated* |

Reasoning, *estimated*.
- With a geometric acceptance model, τ = (1 − α⁸)/(1 − α), so Table 1's chat 3.64 means α ≈ 0.75 (*computed*).
- A sealed target that agrees with dense on 85-92 % of positions of its own text pulls α toward 0.64-0.69 if the
  drafter tracks dense: q_τ 0.74-0.84 (*computed* from those α).
- It pulls α less if the drafter tracks the sealed target through the sealed taps, whose layer 33 already carries
  the sealed model's prediction: q_τ toward 0.95.
- The head is a frozen copy, so the int4 table shifts drafter and target logits together: Δτ near 0, slightly
  positive on near-ties.
- Row 151669 is a padding row the target never trained. Its int4 g64 error relative to its own scale is the largest
  of any row's, hence the lower cosine and the one-sided interval.

## 7. Decision rules

Primary quantity: q_τ for the best neural drafter with the `shared-bf16mask` head, on chat and on code. It is the
ratio of corpus τ (total tokens over total rounds), on odd prompt ids, with a paired bootstrap over prompts (10,000
resamples). The best drafter is the one with the highest sealed corpus τ on even ids, chat and code pooled. Bands use
the point estimate: high ≥ 0.85, mid [0.65, 0.85), low < 0.65.

- **D0, pilot (interim).**
  - q_τ ≥ 0.90 on both chat and code: D1's "both high" is proposed at once. The full run continues for D2-D6 only if
    the operator says so.
  - q_τ ≤ 0.60 on both: D1's "any low" is proposed at once. The full run shrinks to the references, prompt lookup,
    4c and agreement (D3, D5, D7).
  - Otherwise: the full run.
- **D1, neural route.**
  - Both high: no fine-tune is proposed, and neural integration waits only for a flat verify.
  - Any low: the operator is asked whether neural drafting continues at 4B, with a fine-tune pilot (L5, $18-30) as
    the only way to test recovery.
  - Otherwise (both at least mid, not both high): the fine-tune pilot is proposed before any neural integration.
- **D2, head.** On the D1 drafter, Δτ = τ(shared-bf16mask) − τ(own), sealed target, odd ids.
  - Δτ ≥ −0.10 on chat and on code: rule-6 accounting proposes the shared head, 0 head bytes.
  - Else if Δτ ≥ −0.30 on both: both accountings go to the operator.
  - Otherwise: an own head is required (+0.774 b/param in int8), and only an EAGLE-3-class drafter stays inside the
    margin.
- **D2b, mask row.** Δτ = τ(shared) − τ(shared-bf16mask).
  - ≥ −0.05 on chat and on code: the sealed table also serves the mask row.
  - Otherwise: the drafter keeps its BF16 row 151669, 5,120 bytes.
- **D3, zero-byte.** Prompt lookup on loop-truncated extraction text, with the gate and k chosen on even ids and
  scored on odd ids. β_ex comes from L1a at each prompt's mean context:
  β_ex(p) = per_row_exact(p)/t1(p) (`prereg-speculation-0.md` §5). β_B2 subtracts B2's *estimated* savings,
  1.615 ms a row. β_B3 keeps B2's non-KV row cost plus 0.05 ms.
  - L1a has not run: S is reported over the β grid, and no route is proposed.
  - S at β_ex ≥ 1.10: build B1 now and run E0.
  - Else if S at β_B2 ≥ 1.10: B1 and B2 are proposed as one build.
  - Else if S at β_B3 ≥ 1.20 on the 6-8k slice: zero-byte waits for B3.
  - Otherwise: it waits for the brick.
- **D4, quantized drafter.** Tetra stays unprobed whatever these probes show: a Tetra body needs its own probe or a
  QAT plan.
  - int4 probe Δτ/τ ≥ −5 %: an int4 drafter body is licensed.
  - Else if the int8 probe is ≥ −5 %: int8 is the floor.
  - Otherwise: only BF16 or a QAT drafter.
- **D5, loops.**
  - On a corpus where the sealed loop rate is ≥ 10 %, the loop-truncated τ becomes that corpus's primary number.
  - Otherwise the full-text τ stays primary.
  - Prompt lookup always uses loop-truncated text.
- **D6, context window.**
  - The 2,048 window loses ≤ 5 % of τ: it enters the rule-6 table as an option that cuts drafter KV at 8k by 4×.
  - Otherwise drafter KV is counted in full.
- **D7, how well the rebuild stands for the served kernel** (4c only).
  - Top-1 agreement ≥ 0.98: q_τ is read as is.
  - Otherwise: q_τ carries the measured disagreement as a caveat, and math τ on the served references becomes the
    primary math number.
  - 4c not run: q_τ carries the caveat that its reference object was never compared with the served kernel.

Any stop is a proposal; the operator decides.

## 8. Cost

$0. The wall clock is *estimated* on the M3 Max (dense f16 at 8 GB a target, ~400 GB/s by Apple's specification), and
it is informational.
- Pilot: about 1-2 h, mostly references (160 continuations of ~400 tokens at 40-50 tok/s); reused by the full run.
- References: 2 × 141 k tokens at 40-50 tok/s, about 2 h. Prompt lookup: minutes.
- DSpark and DFlash replays: 9 arms × ~47 k rounds × 20-30 ms, about 2.4-3.5 h.
- EAGLE-3: 3 arms × ~54 k rounds × 50-70 ms, about 2.3-3.2 h.
- V1 1-1.5 h; agreement 0.5 h; probes 1 h; thinking 0.5-1 h; V2 0.3 h; 4c about 0.5 h.

In all, 11-14 h over two nights. Downloads: about 15 GB. Peak memory: about 26 GB, both 4B targets in one process
for the agreement stage.

## 9. What this run does not establish

- Any tok/s: the S figures are profiles applied to measured rounds, and every β except L1a's is *estimated*.
- Row-invariance on the card, and t(k) of an exact verify: that is E0.
- The 8B and the 14B: L3, where the 14B needs one target per process.
- That a fine-tune recovers anything: L5.
- What a Tetra-encoded drafter accepts.
- The product's own document distribution beyond the slice the operator supplies.
