# Prereg: the runtime server on Inkling 2 Mini, on one CUDA card, through its OpenAI route (2026-11-24)

DRAFT. Not signed, not stamped. The coordinator fills the predictions, then stamps the file before the run. Once
stamped it is not edited; deviations go in `-ECARTS.md` beside it.

## Question

`llvq serve --local` serves Inkling 2 Mini from local files (`llvq-runtime/src/inkling.rs`). It has never run
against the real model, and its CUDA half is type-checked only. Does it load the model on a card, answer the OpenAI
route in both shapes, make a tool call and read its result, take a first prompt of about 9,000 tokens, and reuse
its cache on a follow-up turn? The operator's Mac is not used for this.

## Object and method

The `llvq` binary of `llvq-runtime`, built with `--features cuda` in `ops/Dockerfile.cuda` (commit of this prereg).
The model is the file of the first card runs: the text checkpoint, the 24 layers of lattice codes, the tokenizer and
the two int4 g64 tables of the sealed file, from `inkling2-mini-inputs-2026-11-22`, each checked at its pin. The
local file names the fused stack (the default), 32,768 positions. Greedy, reasoning effort `none`, as the server's
defaults.

Job `ops/jobs/inkling-serve-card.sh` on an `l40sx1`. Controls: `oracle` MATCH on Qwen3-0.6B; every input at its
pin. Then the server on 127.0.0.1, loaded once by an empty `/api/chat`, and these requests, each saved raw:

- (a) `GET /v1/models`.
- (b) a short chat on `/v1/chat/completions`, not streamed: "What is the capital of France? Answer with one word."
- (c) a streamed chat with `include_usage`.
- (d) one tool, `get_weather(city)`, and "What is the weather in Paris right now?"; then the follow-up with the call
  as returned and a `role: tool` result, streamed. The same pair on `/api/chat`, whose answer carries
  `prompt_eval_cached_count`: the OpenAI route reports no cached count.
- (e) a first prompt of about 9,000 tokens (a long system prompt in 38 sections, 15 tool definitions; 8,837 tokens
  before the template by the checkpoint's tokenizer, *computed*), streamed, its time to first token; the same request
  again; then a follow-up turn on `/api/chat`.
- (f) `nvidia-smi` before the load, after it, after the long prompt, at the end.

Times are wall seconds seen by the client, one run each. Time to first token is the arrival of the first stream
chunk that carries content, reasoning or a call, from just before `curl` starts.

## Decision rule

Correctness checks. One FAIL makes the job exit non-zero, and the server is not ready for a user.

| check | passes when |
|---|---|
| `a-models` | the model's name is listed |
| `load` | HTTP 200 and `done_reason` `load` |
| `ps-loaded`, `ps-end` | `/api/ps` says `ready` on `cuda`, and the server log holds no panic |
| `b-chat` | HTTP 200, `finish_reason` `stop`, the content contains "Paris" |
| `c-stream` | the stream ends with `[DONE]`, no error event, content present, usage with `prompt_tokens` |
| `d1-tool-call` | `finish_reason` `tool_calls`, a call to `get_weather` with an id and a JSON object holding `city` |
| `d2-tool-followup` | streamed answer with content, `finish_reason` `stop`, no new call |
| `n1-native-tool-call`, `n2-followup` | the same pair on `/api/chat` answers |
| `e1-long`, `e2-long` | the long request is answered, with `prompt_tokens` |
| `e3-followup` | the follow-up turn on `/api/chat` is answered |

Reading checks. They say PASS or FAIL and change nothing: `d1-city` (Paris), `d2-uses-result` (the answer names
18), `e1-prompt-size` (8,000 to 10,000 tokens), `n2-reuse-prompt` and `e3-reuse-prompt` (the cached count covers
the first request's prompt), `n2-reuse-answer` and `e3-reuse-answer` (it covers the answer too), `e2-repeat-reuse`
(the repeat's first token comes in under half the first's time).

What the code says before the run, for the coordinator to weigh: a repeated request whose prompt is one segment is
read again in full (`run_inner`, "the whole prompt is held: its last segment is read again"). The 9,000-token
prompt has no assistant turn, so it is one segment, and `e2-repeat-reuse` should FAIL. The follow-up turn `e3` is
the reading of the reuse.

## Signed prediction (the coordinator, before the run)

| quantity | predicted |
|---|---|
| every correctness check PASS | probability 0.3 (a first run of the server on the real model) |
| load, seconds | 30 to 60 |
| (b) seconds to the whole answer | 0.5 to 3 |
| (c) seconds to the first token | 0.3 to 1.5 |
| (d1) a call to `get_weather` with `city` Paris | probability 0.6 |
| (d2) seconds to the first token of the follow-up | 0.2 to 2 |
| (n2) cached count above n1's prompt (the answer reused) | probability 0.5 |
| (e1) prompt tokens | 8,800 to 9,600 |
| (e1) seconds to the first token | 55 to 80 (64.12 s for 9,000 tokens by `inkrun`, *measured* today) |
| (e2) over (e1), time to the first token | 0.9 to 1.1 (a repeat is read again, by design) |
| (e3) tokens read, prompt minus cached | under 300, probability 0.5 |
| GB on the card after the load | 8.3 to 9.5 |
| GB on the card after the long prompt | 15 to 22 |

For scale, *measured* on the same card and stack outside the server (`inkling-cuda-ttft`, 2026-11-24): 2,000
tokens read in 7.36 s with the experts by rows, 15.5 GB at the end of that run, 0.015 s a step.

## Cost (*estimated*)

About 15 minutes on an `l40sx1` at $1.80 an hour, $0.45; 30-minute timeout, $0.90 at most. Setup (oracle, inputs
copied and checked) about 5 minutes, the load 1 to 3, the long prompt twice about 1 to 2, the rest under a minute.
The image must be rebuilt from a commit that carries the server (`ops/run.py publish`), which is not counted here.
