# llvq

`llvq` pulls, runs and serves the sealed LLVQ files with Ollama's commands and API. It also
speaks OpenAI's chat format, streams, and describes itself in an OpenAPI 3.1 document. The
plan, its gates and its open decisions are in [`docs/plan-runtime.md`](../docs/plan-runtime.md).

```bash
cargo build --release -p llvq-runtime --features metal   # macOS; --features cuda on Linux; neither: the CPU path (Tetra codes read on the CPU)
llvq pull qwen3:4b                                       # 1.42 GB from the Hub, checked against its pinned sha256
llvq run qwen3:4b                                        # starts a server if none answers
llvq run qwen3:4b "Name three primary colours."
llvq ls; llvq ps; llvq show qwen3:4b; llvq stop qwen3:4b; llvq rm qwen3:4b
```

| model | file | b/param | Metal config | CUDA config |
|---|---|---|---|---|
| `qwen3:4b`, also `qwen3:latest` | `qwen3-4b-sealed.bin`, 1.42 GB | 2.7320 | expanded | 4E v9 (greedy only) |
| `qwen3:8b` | `qwen3-8b-sealed-B.bin`, 2.82 GB | 2.6953 | expanded, never run fused on Metal | 4E v9 (greedy only) |
| `qwen3:14b` | `qwen3-14b-sealed.bin`, 5.09 GB | 2.7305 | expanded, never run fused on Metal | 4E v9 (greedy only) |

The library is `library.json`: each model pins a Hub commit, a size and a sha256. The served
configs are compiled in from `configs/`. The server listens on `127.0.0.1:11435`, one port above
Ollama's, so both can run on one machine.

## Chat in the browser

`llvq serve` serves a chat page at `http://127.0.0.1:11435/`. It is one HTML file compiled into the
binary, with nothing fetched from another host. It streams through `/api/chat`, stops an answer on
the engine side, shows the model's reasoning apart when `think` is on, and offers to pull a model of
the library when the store is empty.

## Routes

Ollama: `/api/chat`, `/api/generate`, `/api/tags`, `/api/ps`, `/api/show`, `/api/pull`,
`/api/delete`, `/api/copy`, `/api/version`, streamed as NDJSON. OpenAI: `/v1/chat/completions`,
`/v1/completions`, `/v1/models`, streamed as server-sent events ending in `data: [DONE]`. The
document is at `/openapi.json` and committed as `openapi.json`. Embeddings, `create`, `push`,
blobs and `/v1/responses` answer `501` with the reason.

Tool calls are served on `/v1/chat/completions` and `/api/chat` for the qwen3 family, in its
official template's format (`src/tools.rs`): `tools` of type `function`, `role: tool` results,
assistant messages that carry `tool_calls`, and answers with `tool_calls` and, on the OpenAI route,
`finish_reason: tool_calls`. A call is streamed whole in one chunk, with its `index`, once its
closing marker has arrived. `tool_choice` `auto` and `none` are served; `none` does not show the
tools to the model and reads nothing for calls. `tool_choice` `required` or a named function, and
`parallel_tool_calls: false`, are refused: the decode is not constrained.

## Inkling 2 Mini from local files

The Inkling 2 Mini file is not published, so it is not in the library. A small JSON file names its
parts, and `llvq serve --local` registers it in the store under its name, family `inkling2`
(`src/inkling.rs`). Relative paths are read from the file's own directory.

```json
{
  "name": "inkling2-mini:local",
  "family": "inkling2",
  "config": "textonly/config.json",
  "checkpoint": "textonly/model.safetensors",
  "layers": "layers",
  "tokenizer": "native/tokenizer.json",
  "tables": "tables",
  "context": 32768
}
```

`stack` is `fused` by default (the stack `inkgsm --stack fused` measures, which reads the int4
tables) or `candle`. `require_layers` defaults to every layer. On CUDA the experts read a prompt's
rows in one launch a step. The load takes minutes, so keep the model loaded:

```bash
cargo build --release -p llvq-runtime --features metal          # --features cuda on a Linux card
LLVQ_KEEP_ALIVE=-1 target/release/llvq serve --local <data>/inkling.json
LLVQ_HOST=0.0.0.0:11435 LLVQ_KEEP_ALIVE=-1 target/release/llvq serve --local <data>/inkling.json   # a card reached by a tunnel
```

The prompt is the checkpoint's template (`llvq_llm::inkchat`). The reasoning effort is the
request's `reasoning_effort` (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` or a
number in [0, 0.99]); absent, it is `none`, as our GSM8K runs. The reasoning streams in the
`reasoning` field. Calls come back as `tool_calls`, each whole in one chunk, with
`finish_reason: tool_calls`; the arguments a client sends back must be a JSON object. The answer is
greedy unless the request names a temperature. A conversation is read again only from the first
turn that changed: the cache keeps a mark at each turn and rolls back to it. Raw prompts are
refused.

opencode, with the configuration shape its version 1.18.20 reads (`opencode.json`):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "llvq": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "llvq local",
      "options": { "baseURL": "http://127.0.0.1:11435/v1", "apiKey": "local" },
      "models": {
        "inkling2-mini:local": {
          "name": "Inkling 2 Mini, Tetra",
          "tool_call": true,
          "limit": { "context": 32768, "output": 4096 }
        }
      }
    }
  },
  "model": "llvq/inkling2-mini:local",
  "small_model": "llvq/inkling2-mini:local"
}
```

`limit.context` must be the `context` of the local file: opencode assumes 200,000 for a model it
does not know. The engine holds one conversation, so a request from another conversation (a title
that `small_model` asks for, say) makes the next turn of the first read its whole history again.

## Limits

The engine decodes one sequence at a time, so requests to one model queue. On CUDA the three
models run 4E, which decodes greedily: a temperature above 0 is refused there. A chat history is read again
only from the first message that changed since the last request. Rates in a response are the
runtime's own, never a measurement. A call block the model writes that is not a JSON object with a
`name` comes back as content, markers included, with a warning in the unstreamed response. A tool
result is placed by its position after the calls, as Qwen3's template places it; its
`tool_call_id` is checked to name a call of the assistant message before it.
