# llvq

`llvq` pulls, runs and serves the sealed LLVQ files with Ollama's commands and API. It also
speaks OpenAI's chat format, streams, and describes itself in an OpenAPI 3.1 document. The
plan, its gates and its open decisions are in [`docs/plan-runtime.md`](../docs/plan-runtime.md).

```bash
cargo build --release -p llvq-runtime --features metal   # macOS; --features cuda on Linux; neither: dense CPU path
llvq pull qwen3:4b                                       # 1.42 GB from the Hub, checked against its pinned sha256
llvq run qwen3:4b                                        # starts a server if none answers
llvq run qwen3:4b "Name three primary colours."
llvq ls; llvq ps; llvq show qwen3:4b; llvq stop qwen3:4b; llvq rm qwen3:4b
```

| model | file | b/param | Metal config | CUDA config |
|---|---|---|---|---|
| `qwen3:4b`, also `qwen3:latest` | `qwen3-4b-sealed.bin`, 1.42 GB | 2.7320 | expanded | 4E v4 (greedy only) |
| `qwen3:8b` | `qwen3-8b-sealed-B.bin`, 2.82 GB | 2.6953 | expanded, never run fused on Metal | grouped |
| `qwen3:14b` | `qwen3-14b-sealed.bin`, 5.09 GB | 2.7305 | expanded, never run fused on Metal | grouped |

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

## Limits

The engine decodes one sequence at a time, so requests to one model queue. The 4B on CUDA runs
4E, which decodes greedily: a temperature above 0 is refused there. A chat history is read again
only from the first message that changed since the last request. Rates in a response are the
runtime's own, never a measurement.
