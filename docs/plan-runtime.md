# Plan: an Ollama-style runtime for the sealed files

Goal: one binary, `llvq`, that pulls a sealed file, runs it and serves it over HTTP with Ollama's commands and API on
every platform the engine supports. The API also speaks OpenAI's chat format, is described by an OpenAPI 3.1 document,
and streams. Nothing is run until the operator gives a go on each stage (rule 1). Stages R0 to R4 and R6 cost 0 $ and
run on the Mac. R5 needs one card.

## Why

Today a stranger who wants to talk to a sealed file needs the repository, `cargo`, the right feature flags and a
served config path (`bin/chat`), or a Mac and the one-binary bundle of `ops/metal-package/`. Neither has a model
library, an HTTP API or a stable command set. Ollama's command set and API are what local LLM clients already speak.
Matching them puts the 2.7 b/param files behind tools people already use, with no Python and no repository.

## Facts the plan rests on

### The engine, read in this repository on 2026-11-06

- `llvq_llm::repl` already holds the four pieces a chat needs and `generate` lacks: stop on `<|im_end|>`, sample
  (temperature and nucleus, seeded SplitMix64), stream with partial UTF-8 held back, and keep the KV cache across
  turns. `bin/chat` and `bin/metalrun chat` share it.
- `llvq_llm::chatfmt` builds Qwen3's chat format from token ids, with the empty reasoning block when thinking is off.
  The markers come from the tokenizer the sealed file carries.
- `served::load_resolved` loads a sealed file through the fused backend of the build. `Served::serve` applies a
  config's attention form, store and decode engine, and refuses what it does not serve on the device.
- `KvCache::set_len` cuts a cache back to a shorter prefix. The REPL uses it after a 4E launch.
- The engine runs batch 1. `Qwen3::hidden_cached` takes one sequence, and the caches belong to that sequence.
- The model type is `Qwen3`. No other architecture loads.
- The persistent decode (4E) takes the argmax on the card. `repl::check_served_chat` refuses a temperature above zero
  under it. 4E decodes at most 64 tokens a launch and 8,704 positions (`served_decode.rs`), on the 4B's shape only.
- On Metal, `Served::check_device` refuses the grouped and kernel forms and 4E. Only an expanded config serves there.
- `load_resolved` has two arms, CUDA on Linux and Metal on macOS. Any other build bails and points to the dense path,
  `sealed::load`. `llvq-cuda` is target-gated to Linux.
- The CUDA kernels are compiled by NVRTC at launch, so a CUDA target needs the NVRTC library present.
- A sealed file opens with no checkpoint and no network: it carries its tokenizer and its `config.json`.
- `tokio`, `hyper`, `tower` and `reqwest` are already in `Cargo.lock`, pulled by `hf-hub`. `axum` is not.

### The objects

| name proposed | Hub repository | file | bytes | sha256 | config on CUDA | config on Metal |
|---|---|---|---|---|---|---|
| `qwen3:4b` | `Pier-Jean/Qwen3-4B-LLVQ-Tetra-sealed` | `qwen3-4b-sealed.bin` | 1,418,224,685 | `886391a8…` | `qwen3-4b-tetra-e4-4e-v4.json` | `qwen3-4b-tetra-e4.json` |
| `qwen3:8b` | `Pier-Jean/Qwen3-8B-LLVQ-Tetra-sealed` | `qwen3-8b-sealed-B.bin` | 2,815,098,745 | `7bdb9a55…` | `qwen3-8b-tetra-e4-grouped.json` | `qwen3-8b-tetra-e4.json` |
| `qwen3:14b` | `Pier-Jean/Qwen3-14B-LLVQ-Tetra-sealed` | `qwen3-14b-sealed.bin` | 5,087,000,541 | `61db37fe…` | `qwen3-14b-tetra-e4-grouped.json` | `qwen3-14b-tetra-e4.json` |

Sizes and digests come from `configs/README.md`. The full digests go in the manifest, not here. The three sealed
repositories are frozen because paper 2 cites them, so a pinned revision never moves under the runtime. The Metal
journals cover the 4B only: the 8B and 14B have not run fused on Metal.

### Ollama, read in `ollama/ollama` on GitHub, branch `main`, on 2026-11-06

Sources: `docs/api.md`, `docs/openapi.yaml`, `docs/cli.mdx`, `docs/api/openai-compatibility.mdx`. The revision is
pinned in stage R2, where the contract test reads it.

- Native API: `POST /api/generate`, `POST /api/chat`, `GET /api/tags`, `GET /api/ps`, `POST /api/show`,
  `POST /api/pull`, `DELETE /api/delete`, `POST /api/copy`, `GET /api/version`, `POST /api/create`, `POST /api/push`,
  `POST /api/embed`, `HEAD|POST /api/blobs/{digest}`.
- Ollama publishes its own OpenAPI document, `docs/openapi.yaml`, version `3.1.0`.
- Streaming is on by default (`"stream": true`). The wire format is `application/x-ndjson`, one JSON object a line.
- The last chunk carries `done`, `done_reason`, `prompt_eval_count`, `eval_count` and durations in nanoseconds.
- Model names are `name:tag`, with `latest` as the default tag. The examples use port 11434.
- CLI: `run`, `pull`, `rm`, `ls`, `ps`, `stop`, `serve`, `create`.
- OpenAI compatibility: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/models/{model}`,
  `/v1/embeddings`, `/v1/responses`. The page does not name the stream transport. OpenAI's own API streams as
  server-sent events ending in `data: [DONE]`, and its clients expect that.

## Design

### Where it lives

A new workspace crate, `llvq-runtime`, builds the binary `llvq`. It depends on `llvq-llm` and carries its `cuda` and
`metal` features through. The five core crates do not move and take no dependency. New dependencies stay in this
crate: `axum` for HTTP, `clap` for the command line, `utoipa` for the OpenAPI document (decision D4), `sha2` for the
digests.

The precedent is `llvq-tetra/`: a distribution that ships the served kernels stays in this repository, because the
kernels and the format move with the engine.

### The model library

The runtime keeps its files under `LLVQ_HOME`, default `~/.llvq/`, laid out like Ollama's store:

```text
~/.llvq/blobs/sha256-<hex>             the sealed file, named by its digest
~/.llvq/manifests/<name>/<tag>.json    name:tag -> blob, size, Hub repo and revision, config per backend
```

The library itself is a file in the repository, `llvq-runtime/library.json`, compiled into the binary. It maps each
`name:tag` to a Hub repository, a pinned commit, a file name, a size, a full sha256 and a config per backend. The
config files are embedded from `configs/` with `include_str!`, so the runtime serves the bytes the journals pinned.

`llvq pull` downloads with `hf-hub`, already a dependency, and checks the size then the sha256 before the manifest is
written. A digest that does not match leaves nothing registered. `llvq rm` removes the manifest, then the blob when
no manifest names it. `llvq cp` writes a second manifest on the same blob.

### The engine worker

Each loaded model is owned by one OS thread. That thread holds the `Qwen3`, its `Decoder` and the caches of the last
conversation, and nothing else touches them. The HTTP side sends jobs on a channel and receives tokens on another.
This keeps the candle and CUDA objects off the async runtime, so their `Send` and `Sync` bounds never come up.

The engine is batch 1, so requests to one model run one after another, in arrival order. `GET /api/ps` shows the
queue. Batching several sequences is out of scope: the engine has no batched decode.

Ollama's chat API is stateless: every request carries the whole history. Re-reading it every turn would cost the
prefill again, about 3.929 ms a token through the served CUDA kernel (`f1e0-2026-09-10`, as quoted in
`bin/chat.rs`) and 31.5 s for 4,096 tokens on the M3 Max (*measured*,
[metal-nuit](mesures/metal-nuit-2026-11-06.txt)). So the worker keeps the token ids of the cached conversation. A new
request keeps the longest common prefix with `KvCache::set_len` and prefills only the rest.

A client that disconnects sets a flag. The worker reads it between tokens on the eager path and between launches
under 4E, so a cancelled request stops within one token or one launch of 64.

The generation loop moves from `repl.rs` into a library type in `llvq-llm`, a `Session`: feed ids, generate with a
sampler, stop tokens and a callback per token, cancel. `repl.rs` is rewritten on top of it, so the REPL and the
server run one loop.

### The API

| route | phase | streams | notes |
|---|---|---|---|
| `GET /api/version` | R2 | no | runtime version and the commit it was built from |
| `GET /api/tags` | R2 | no | models in the library |
| `GET /api/ps` | R2 | no | loaded models, backend, config, queue length |
| `POST /api/show` | R2 | no | manifest, config, b/param from `configs/README.md`, backend |
| `POST /api/generate` | R2 | NDJSON | raw prompt or templated, per Ollama's `raw` field |
| `POST /api/chat` | R2 | NDJSON | roles `system`, `user`, `assistant`; Qwen3 template from `chatfmt` |
| `POST /api/pull` | R2 | NDJSON | download progress, then the digest check |
| `DELETE /api/delete`, `POST /api/copy` | R2 | no | the store |
| `GET /v1/models`, `GET /v1/models/{model}` | R3 | no | OpenAI shape |
| `POST /v1/chat/completions`, `POST /v1/completions` | R3 | SSE | OpenAI shape, `data: [DONE]` at the end |
| `GET /openapi.json` | R3 | no | the OpenAPI 3.1 document of this server |
| `/api/embed`, `/api/embeddings`, `/api/create`, `/api/push`, `/api/blobs/*`, `/v1/embeddings`, `/v1/responses` | R2 | no | `501` with a JSON error naming the reason |

Request options map to the engine as follows. `temperature`, `top_p`, `seed`, `num_predict` and `stop` reach the
sampler and the stop set. `num_ctx` above the model's limit is refused with `400`. The limit is RoPE's
`max_position_embeddings`, or 8,704 under 4E. Options the sampler does not implement, such as `top_k` or
`repeat_penalty`, follow decision D3. Thinking maps to `chatfmt::open_assistant`; the field name follows Ollama's
document, checked in R2.

The last chunk reports `eval_count`, `prompt_eval_count` and durations as Ollama does. Those rates are the runtime's,
taken with no warmup and no rounds. They are never quoted as measurements (rule 8). Published speeds stay with
`fusedrun` and `planesbench`.

### The OpenAPI document

The document is generated from the handler types with `utoipa`, so the spec and the code cannot drift. A copy is
committed at `llvq-runtime/openapi.json`, and a test fails when the generated document differs from it. A second test
reads Ollama's `docs/openapi.yaml` at a pinned commit and checks that every `/api` route we answer exists there with
the same method and the same required request fields. Ollama is MIT-licensed, so the pinned copy can sit in the
crate's test fixtures with its notice.

### The command line

```text
llvq serve                       start the server on LLVQ_HOST
llvq pull qwen3:4b               download, check size and sha256, register
llvq run qwen3:4b                interactive chat, /bye and /reset as in bin/chat
llvq run qwen3:4b "a prompt"     one answer, streamed, then exit; stdin is read when piped
llvq ls | ps | show | stop | rm | cp
```

The client commands talk HTTP to `LLVQ_HOST`, as Ollama's do. Whether `run` starts a server when none answers is
decision D7. Every new variable (`LLVQ_HOME`, `LLVQ_HOST`, `LLVQ_KEEP_ALIVE`, `LLVQ_MAX_LOADED`) goes into the table of
`CLAUDE.md` in the stage that adds it.

## Platforms

The commands are the same everywhere. The engine behind them is not, and `llvq ps` and `llvq show` print which one
runs.

| platform | backend | served config | state |
|---|---|---|---|
| Linux x86_64, NVIDIA card | fused CUDA | 4E v4 at 4B, grouped at 8B and 14B | served door *measured* on one L40S only; a consumer card is unmeasured (`ETAT.md` §5) |
| macOS, Apple silicon | fused Metal | expanded configs only | 4B *measured* on one M3 Max; 8B and 14B never run fused on Metal |
| Linux or macOS, no accelerator | dense, `sealed::load` | none | builds today; speed and memory on CPU not measured |
| Windows | none | none | not built, not tested; `llvq-cuda` is gated to Linux |

## Stages

| stage | content | gate | cost |
|---|---|---|---|
| R0 | crate, `library.json`, store, `pull` with digest check, `ls`, `rm`, `show`, `cp` | the 4B pulled from the Hub reads sha256 `886391a8…`; a mutated digest in the manifest is refused and leaves no manifest | 0 $, Mac, 1.42 GB download |
| R1 | `Session` in `llvq-llm`, `repl.rs` rewritten on it, prefix reuse | `bin/chat` at temperature 0 prints the same ids as before the move (`LLVQ_CHAT_IDS=1`, Metal fused 4B); a two-turn chat replayed statelessly gives the stateful run's ids; the existing `repl` tests pass unchanged | 0 $, Mac |
| R2 | server, Ollama routes, NDJSON, queue, cancellation, `501` routes | greedy `/api/chat` streamed gives `bin/chat`'s ids; a disconnect stops the worker within one token; the contract test against Ollama's `openapi.yaml` passes; the official `ollama` Python client streams a chat unmodified | 0 $, Mac |
| R3 | OpenAI routes with SSE, `utoipa` document at `/openapi.json` | the official `openai` Python client streams a chat unmodified; the committed document equals the generated one, and a route added without its annotation fails the test | 0 $, Mac |
| R4 | `run`, `ps`, `stop`, pull progress in the CLI | a scripted session pulls, runs one prompt, lists, stops and removes, on a clean `LLVQ_HOME` | 0 $, Mac |
| R5 | the same gates on CUDA, under the served 4B config, then 8B and 14B | greedy `/api/chat` gives the ids of `bin/chat` under the same config; the decision D2 behaviour holds; the bin is in `cargo build --bin` and the runtime `COPY` of `ops/Dockerfile.cuda`, and in `UPLOAD_ALLOW` of `ops/run.py` | under 1 $ (*estimated*: `l40sx1` at 1.80 $/h, 20 to 30 billed minutes, as the 8 to 16 minutes of the 2026-11-05 jobs) |
| R6 | release binaries, one per platform build, and a short README | a binary copied to a clean machine pulls and answers with no repository and no compiler | 0 $ |

Before any commit that touches `llvq-llm` (R1), the full suite runs in release with the ignored tests (rule of
`CLAUDE.md`). `cargo clippy --all-targets` stays at zero warnings at every stage.

## Decisions for the operator

- **D1. Names.** Binary `llvq`, crate `llvq-runtime`, tags `qwen3:4b`, `qwen3:8b`, `qwen3:14b`, with `latest` on the
  4B. The default port is proposed at 11435: 11434 is the port Ollama's examples use, and both servers may run on one
  machine.
- **D2. Sampling under 4E.** The served 4B config decodes greedy only, and most chat clients send a temperature.
  Three ways: refuse with `400` naming the reason, which matches the repository's refusal rule; decode such requests
  on the eager engine with the same weights and the same attention form, which is a combination no config names; or
  load the `kernel-from1280` config for sampled requests, which costs a second copy of 1.38 GB. Proposed: refuse,
  until the operator names a config for sampling.
- **D3. Options the sampler does not implement.** Ignored with a warning in the response, or refused. Proposed:
  ignored and listed in a `warnings` field, because clients send them by default.
- **D4. OpenAPI from the code (`utoipa`) or written by hand.** Proposed: from the code, with the committed copy and
  its test.
- **D5. Metal configs.** The runtime serves on Metal what `served.rs` serves there, the expanded form. Lifting the
  grouped form on Metal is already an open decision of `ETAT.md` §5, and the runtime follows it.
- **D6. The dense CPU path.** Exposed under the same commands, with `ps` saying "dense", or refused. Proposed: exposed,
  since it is the only path on a machine without a GPU, with its speed unmeasured and said so.
- **D7. `run` with no server.** Print how to start one, or start one in the background. Proposed: start one, as a
  desktop user expects.

## Out of scope

Embeddings, `create` and the Modelfile, `push`, images, tool calls, architectures other than Qwen3, batched decoding,
and Windows. Each is answered `501` or refused by name rather than half-implemented.

## Limits a user will meet

Two clients on one model wait for each other: the engine is batch 1. The 4B under 4E holds at most 8,704 positions. A
4,096-token pasted prompt takes about 31.5 s to read on the M3 Max before the first answer token. On CUDA the target
needs the NVRTC library, because the kernels are compiled at launch.
