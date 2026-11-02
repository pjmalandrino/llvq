# Prereg vllm-tetra: the sealed 4B served by vLLM, against our engine and against AWQ (draft)

Status: draft, not timestamped, no go. It is timestamped and committed before the first measurement, or not run.
Cost: about **$0.55** on `l40sx1`, *estimated* below, timeout 45 min, so **$1.35** at worst. No cap is in force
(`docs/ETAT.md` §5); one is owed before launch. Spent so far: $246.75 over 227 priced jobs on `main`, more on branch
`debit-trims` (*measured*, `docs/data/jobs.csv`); the total is read again when this file is timestamped.
Measured code: the commit that carries `llvq-vllm/` and the four ops of `llvq-tetra/llvq_tetra/csrc/tetra_cuda.cu`.
Design: [`docs/plan-vllm.md`](../docs/plan-vllm.md).

## Question

Does vLLM 0.26.0, through `llvq-vllm`, compute what our engine computes on the sealed Qwen3-4B, within the tolerance of
`docs/ETAT.md` §4? And how fast does it decode at batch 1, and reach the first token after 8,192 tokens, against AWQ in
the same image? No answer exists: the plugin never ran on a card, and the four ops never compiled. It decides whether
vLLM becomes the delivery path for speed, and it gives the first speed comparison of Tetra and AWQ inside one stack.

## Setup

One job, `l40sx1`, image `vllm/vllm-openai:v0.26.0` at digest `ffb2d59b…`, launched as `ops/jobs/a100-vllm.sh` is:
`uv run ops/run.py bench --image … --timeout 45m --bucket Pier-Jean/jobs-artifacts`. The job script
`ops/jobs/vllm-tetra.sh` and its driver `ops/vllm_tetra.py` are still to write; this draft fixes what they do.

Objects. Tetra: `Pier-Jean/Qwen3-4B-LLVQ-Tetra` at a revision pinned in the timestamped version, whose `config.json`
names `artifact_sha256` `886391a8…`. AWQ: `Qwen/Qwen3-4B-AWQ` at `74d4bd2b…`. Tokenizer for both: `Qwen/Qwen3-4B` at
`1cfa9a72…`. Engine arguments as `ops/awq_speed.py`: `dtype="float16"`, `enable_prefix_caching=False`, `seed=0`,
`max_model_len=8448`; `gpu_memory_utilization` 0.30 for Tetra, whose prompt log-probabilities take about 2.5 GB at a
time, and 0.13 for AWQ. Both arms live in one process.

1. `pip install --no-deps` of `llvq-tetra` and `llvq-vllm` from the bucket; build the CUDA extension, tile 64.
2. Per-row checks on the card against `PackedModel.dequantize`, one record of each shape: `rot_apply_rows` then
   `tv_tetra48` on layer 0's `q_proj` and `gate_proj` and its `down_proj` (odd factor 19); `tv_q4` on `v_proj`, `o_proj`
   and layer 12's `down_proj`; `tv_emb_q4` on three hidden states against the dequantized table.
3. Load both arms. Read back `quantization` from each engine's config, the layers our methods armed, and their bytes.
4. **Gate 1a.** Prompt ids `[785, 6722, 315, 9625, 374]`, 256 greedy tokens (`temperature=0`, `ignore_eos`,
   `min_tokens=max_tokens=256`), the sha256 of the decoded text against `0597cee8…`, our engine's line
   (`served-grouped-2026-11-02`, branch `debit-trims`).
5. **Gate 1b.** The 12 windows of 4,096 tokens of WikiText-2 test, token fingerprint `3f1baca9033bf251`, each with
   `prompt_logprobs=0`, `max_tokens=1`; NLL over positions 1 to 4,095 as `window_nll` scores them. The paired 95 % t
   interval of `ppl_vllm / ppl_engine − 1` over the windows, against the published arm's prefill NLLs of
   `gqa-ppl-2026-11-01` (12.5822 in all). The same 12 windows on the AWQ arm, for control 5.
6. **Gate 2, graphs on**, vLLM's default. Interleaved rounds as `awq_speed.py`: 2 discarded, 5 timed, 128 tokens after
   the 5-token prompt, prefill included. Then the first 8,192 tokens of the corpus as one prompt: time to the first
   token (`max_tokens=1`), and 256 decoded tokens after it, 1 discarded and 5 timed rounds, interleaved.
7. **Gate 2, eager**, both arms reloaded with `enforce_eager=True`, step 6 again. Secondary: it answers the rule 3
   question of `docs/plan-vllm.md` and is never the product number.

## Controls, each one voids the run

1. The card reads L40S; `vllm 0.26.0` is read in the installed package; the image digest is the pinned one.
2. `torch.__version__` reads 2.11.0 after the install: the image's torch did not move.
3. The extension builds, and all seven per-row checks lie under 1e-2, the bar of the CUDA stage.
4. `quantization` reads `llvq` on the Tetra arm; 144 linear layers and the embedding are armed, none dense; their
   bytes lie within 1 % of the 1.378 GB computed in `docs/plan-vllm.md`.
5. **The harness.** The AWQ arm's perplexity over the 12 windows lies within 1 % of 13.5207, the A4 figure on the same
   tokens (`a4-campagne-4b-ppl-BRUT-2026-08-06`). A scorer that read a shifted position would miss by far more.
6. Every timed round of an arm returns the tokens of its first round.

## What gets published, and what does not get compared

Published: the per-row errors; the gate 1 verdict with its interval and the first divergent token, if any; per arm the
median tok/s and its range, the time to first token at 8,192 and the decode tok/s after it; the Tetra over AWQ ratio
formed round by round inside each round (rule 7); the measured resident bytes in b/param for the whole model.

Not compared: no tok/s of vLLM against any tok/s of our engine (rule 5); no ratio of this job divided by a ratio of
ours; no eager figure presented as the served speed; nothing about 8B, 14B, batch above 1 or another card.

## Decision rule

"Inside" means both bounds within [−0.12 %, +0.12 %]. An interval not inside either excludes zero or contains it.

| gate 1a, 256 tokens | gate 1b, paired interval | verdict |
|---|---|---|
| identical | inside | tolerance met: the vLLM path serves our file, and gate 2 is published |
| differs | inside | not settled: operator decision; the first divergent token and its logit gap in our engine are reported |
| identical or not | not inside, excludes zero | not met: the path changes quality beyond noise, gate 2 is not published as Tetra's |
| identical or not | not inside, contains zero | lacks power: the operator chooses more windows or a stop |
| otherwise, a step that did not run | | not settled, operator decision |

Four test values: +0.03 % [−0.02, +0.08] falls in row 1 with identical tokens and in row 2 without; +0.20 %
[+0.15, +0.25] in row 3; +0.05 % [−0.03, +0.13] in row 4 whatever the tokens.

## Signed predictions

| quantity | point | interval |
|---|---|---|
| worst per-row error over the seven checks | 4e-4 | [2e-4, 8e-4] |
| resident bytes of our layers | 1.378 GB | [1.36, 1.40] |
| gate 1a, 256 tokens identical | yes | |
| gate 1b, `ppl_vllm / ppl_engine − 1` | +0.02 % | interval inside [−0.10 %, +0.12 %] |
| AWQ perplexity, control 5 | 13.52 | [13.40, 13.65] |
| Tetra over AWQ tok/s at batch 1, graphs on | 0.90 | [0.70, 1.10] |
| time to first token at 8,192, Tetra / AWQ | 1.0 s / 0.6 s | [0.6, 1.6] / [0.4, 0.9] |

Reasoning, *estimated*. The row path runs our kernels, so decode should agree to rounding; the dense prefill rounds to
f16 as vLLM's FP16 arm does. On our bench the 252 projections take 3.734 ms in Tetra against 3.263 in AWQ
(`banc-252-2026-10-06`), while our int4 head reads 0.22 GB where AWQ's f16 head reads 0.78. Known flaw: the time to
first token is mostly the torch dequantization of the prototype, so it says little about the format until the CUDA
dequantization kernel exists.

## Cost

*Estimated* on `l40sx1` at $1.80/h. Start, install and extension build, 4 min. Per-row checks, 2 min. Both loads with
compilation and graph capture, 5 min. Gates 1a and 1b, 2 min. Gate 2 with graphs, 2 min. The eager reload and gate 2
again, 4 min. In all about 18 min, **$0.55**. Timeout 45 min, **$1.35** at worst. A zero-cost probe in the same image on
`cpu-upgrade` first, about $0.01, would catch an import failure before a card is billed.
