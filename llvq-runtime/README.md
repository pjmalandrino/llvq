# llvq

`llvq` pulls, runs and serves the sealed LLVQ files with Ollama's commands and API. It also
speaks OpenAI's chat format, streams, and describes itself in an OpenAPI 3.1 document. The
plan, its gates and its open decisions are in [`docs/plan-runtime.md`](../docs/plan-runtime.md).

## Install

One command installs `llvq` with the CUDA libraries it needs, from the
[latest release](https://github.com/pjmalandrino/llvq-tetra/releases/latest). Nothing else is built or installed:

```bash
curl -fsSL https://github.com/pjmalandrino/llvq-tetra/releases/latest/download/install.sh | sh
llvq run inkling2:mini
```

`install.sh` reads the system and refuses, with the reason, what the archive cannot run on:

| system | archive | the machine must already have |
|---|---|---|
| Linux x86_64 | `llvq-<version>-linux-x86_64-cuda.tar.gz`, the binary and its CUDA 12.4 libraries (cuBLAS, cuBLASLt, cuRAND, NVRTC, cudart) | an NVIDIA driver 550 or newer; a card of compute capability 8.9 or newer (RTX 40xx and 50xx, L4, L40S, H100); glibc 2.35 and libssl3 (Ubuntu 22.04, Debian 12 or later) |
| macOS arm64 | `llvq-<version>-macos-arm64-metal.tar.gz`, the binary alone | Apple Silicon |
| Linux x86_64, no NVIDIA card | `llvq-<version>-linux-x86_64-cpu.tar.gz`, the binary alone | glibc 2.35 and libssl3; AVX2 and FMA for the SIMD path, a scalar path otherwise |
| Linux arm64 | `llvq-<version>-linux-arm64-cpu.tar.gz`, the binary alone | glibc 2.35 and libssl3 |
| macOS x86_64 | `llvq-<version>-macos-x86_64-cpu.tar.gz`, the binary alone | an Intel Mac |

The CPU archives serve the Qwen3 models and Inkling 2 Mini. Inkling's experts stay in their lattice codes and its
two tables in int4; its attention stays in BF16. It decodes at 4.24 tok/s in 11.7 GB on 8 vCPU of an EPYC 7R13
(*measured*, one run, `docs/mesures/inkling-cpu-2026-11-25.txt`). On Linux
x86_64 `install.sh` takes the CPU archive when no usable NVIDIA card answers; `LLVQ_FLAVOR` names one instead.
`LLVQ_CPU_WEIGHTS=nibbles` on the server decodes the Tetra codes once at load: the same tokens, 1.29 to 1.45 times
the rate on the 4B for 0.8 GB more (*measured*, 8 vCPU of an EPYC 7R13, `docs/mesures/llvq-cpu-x86-2026-11-25.txt`).

The CUDA toolkit is not needed to run, and an installed one does not interfere: the binary finds its own libraries
first (an rpath of `$ORIGIN/lib`). A toolkit 13.x beside it is fine, and so is a driver newer than 550. The install
goes to `~/.local/lib/llvq` with a link in `~/.local/bin`, or `/usr/local` as root, or `LLVQ_PREFIX`. Models go to
`~/.llvq` (`LLVQ_HOME`). The archives are built by `.github/workflows/llvq-archives.yml`, run by hand; the Linux
one by `ops/release/build-linux-cuda.sh` in `nvidia/cuda:12.4.1-devel-ubuntu22.04`, which anyone with Docker can
run the same way:

```bash
docker run --rm -v "$PWD":/src -w /src nvidia/cuda:12.4.1-devel-ubuntu22.04 bash ops/release/build-linux-cuda.sh
```

## Build from source

- **Rust through rustup, not the distribution's cargo.** `rust-toolchain.toml` pins the compiler, and only the
  rustup proxies read that file. Ubuntu's packaged cargo ignores it, and is too old for `Cargo.lock` (version 4
  needs cargo 1.78 or newer): it fails with "lock file version 4 requires `-Znext-lockfile-bump`".
- **On Linux, a CUDA toolkit with `nvcc`, version 12.0 to 12.6, 12.8 or 12.9.** The build needs it: candle compiles
  its CUDA kernels with `nvcc` at build time, and cudarc reads the toolkit version from `nvcc --version` to choose
  its bindings and the libraries it links. The toolkit the card jobs build with is 12.4. **A 13.x toolkit does not
  build today**: cudarc 0.19.8, the version in `Cargo.lock`, knows 13.0 to 13.3 and stops on any other version
  ("Unsupported cuda toolkit version: 13.5"), and candle 0.9.2 has never been built against CUDA 13 here. Install
  a 12.x toolkit beside it (`/usr/local/cuda-12.4`, first in `PATH`), or build in the Docker image above.
- **`CUDA_COMPUTE_CAP`** when the build machine has no GPU (`89` for Ada). With a GPU, candle reads it from
  `nvidia-smi`. The binary then runs on that capability and newer. Our own kernels are compiled at run time by NVRTC,
  not at build time.
- **To run**, the NVIDIA driver (550 or newer for a 12.4 build) and the toolkit's libraries on the loader path.

```bash
cargo build --release -p llvq-runtime --features metal   # macOS; --features cuda on Linux; neither: the CPU path (Tetra codes read on the CPU)
llvq pull qwen3:4b                                       # 1.42 GB from the Hub, checked against its pinned sha256
llvq run qwen3:4b                                        # starts a server if none answers
llvq run qwen3:4b "Name three primary colours."
llvq ls; llvq ps; llvq show qwen3:4b; llvq stop qwen3:4b; llvq rm qwen3:4b
```

| model | file | b/param | Metal config | CUDA config |
|---|---|---|---|---|
| `qwen3:4b`, also `qwen3:latest` | `qwen3-4b-sealed.bin`, 1.42 GB | 2.7320 | expanded | 4E v9 after the chain prefill (`LLVQ_PROFILE=eager`: the eager prefill), greedy only |
| `qwen3:8b` | `qwen3-8b-sealed-B.bin`, 2.82 GB | 2.6953 | expanded, never run fused on Metal | 4E v9 (greedy only) |
| `qwen3:14b` | `qwen3-14b-sealed.bin`, 5.09 GB | 2.7305 | expanded, never run fused on Metal | 4E v9 (greedy only) |

`inkling2:mini` (also `inkling2`, `inklingmini-2-tetra`) is a set of files, not one sealed file. The library
pins an index, `llvq-files.json`, which pins every file by size and sha256 (`ops/inkling/files_index.py` writes it):
the text-only checkpoint, the tokenizer, the two int4 tables, and one tar archive per layer of lattice codes. The
pull fetches four files at once (`LLVQ_PULL_JOBS`, `1` for one after another), checks each one, unpacks them under
`~/.llvq/models/` and writes the local file below, so `llvq run inkling2:mini` serves it as `llvq serve --local`
does.

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

The Inkling 2 Mini files are not published, so `llvq pull` refuses them. A small JSON file names its
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

By default the engine decodes one sequence at a time, so requests to one model queue. Under
`LLVQ_BATCH=<n>[:<positions>]` on CUDA, `llvq serve` batches continuously through each model's
`*-4e-v9-batch.json` config: a lone request decodes on 4E, and up to n step together through the
batched GEMM, greedy only, with no conversation cache. `LLVQ_BATCH` is refused on Metal and the CPU.
On CUDA the three models run 4E, which decodes greedily: a temperature above 0 is refused there.
The 4B's chain prefill compiles two tree modules on a card's first start, then loads them from
`LLVQ_MODULE_CACHE` (unset: `$XDG_CACHE_HOME/llvq/modules`, else `~/.cache/llvq/modules`). A chat history is read again
only from the first message that changed since the last request. Rates in a response are the
runtime's own, never a measurement. A call block the model writes that is not a JSON object with a
`name` comes back as content, markers included, with a warning in the unstreamed response. A tool
result is placed by its position after the calls, as Qwen3's template places it; its
`tool_call_id` is checked to name a call of the assistant message before it.
