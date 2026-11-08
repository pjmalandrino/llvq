# Environment variables

Every `LLVQ_*` name the Rust code reads, one row each. `llvq-llm/tests/env_registry.rs` fails when a quoted `"LLVQ_*"`
literal in a tracked `.rs` file is missing from this page, so a new variable cannot ship undocumented. `CLAUDE.md`
keeps the ones an agent needs to resume; this page is the complete list.

Categories: **served** (a config field, the runtime, or a variable refused beside `LLVQ_CONFIG`), **tool** (a knob of
an encoding or evaluation binary), **mode** (turns a binary into a bench of one thing), **test** (a test fixture or
archive location). A served config refuses a variable that contradicts it, every name of `BELOW_THE_DOOR`, and, beside
the persistent decode, every name of `MEASUREMENT_MODES` (`llvq-llm/src/served.rs`). `bin/fusedrun` refuses two of its
modes set together (`ALONE` in that file).

## Served objects and the runtime

| variable | read by | values and default | category |
|---|---|---|---|
| `LLVQ_CONFIG` | `served.rs`; `hcapture`, `hratio` refuse it | path to `configs/*.json`; unset = bench | served |
| `LLVQ_FUSED_LAYOUT` | `fused.rs` | `planes14` (default), `planes12x`, `slot32`, `golay70`, `tetra48` | served (config `layout`) |
| `LLVQ_EMBED` | `fused.rs` | `f16` (default), `q8`, `q4` | served (config `embed`) |
| `LLVQ_ROT_SHARE` | `rotplan.rs` | `0`, `1` | served (config `rot_share`) |
| `LLVQ_FUSE` | `fused.rs` | `0`, `1` | served (config `fuse`) |
| `LLVQ_KV` | `kvq.rs` | `f16` (default), `q8` | served (config `kv`) |
| `LLVQ_KERNEL_DIR` | `fused.rs`, `fused_cuda.rs`, `llvq-cuda`, `planesbench` | directory of kernel sources read instead of the built-in text | served, refused beside a config |
| `LLVQ_TILE_BLOCKS` | `llvq-cuda/src/tile.rs` | unset or `auto` = the card's row; power of two in 32..=512 | served, refused beside a config |
| `LLVQ_PREFILL` | `llvq-cuda/src/tile.rs` | `rows:tile` of the prefill kernel, e.g. `8:64`; unset = served pair | mode, refused beside a config |
| `LLVQ_METAL_KERNEL` | `fused_metal.rs` | `lut` (served, default), `c3`, `ar`, `u2`, `ilp`, `plain`; anything else refused | mode, refused beside a config |
| `LLVQ_NVRTC_ARCH` | `llvq-cuda/src/gpu.rs` | `compute_NN`, default `compute_89` | served (names the card, never refused) |
| `LLVQ_HOME` | `llvq-runtime` store | directory, default `~/.llvq` | served |
| `LLVQ_HOST` | `llvq-runtime` client and server | `host:port`, default `127.0.0.1:11435` | served |
| `LLVQ_KEEP_ALIVE` | `llvq-runtime` engine | duration, default `5m`; `0`; negative = never | served |
| `LLVQ_MAX_LOADED` | `llvq-runtime` engine | count, default `1` | served |
| `LLVQ_BUILD_COMMIT` | `llvq-runtime`, compile time | set by `llvq-runtime/build.rs` | build |

## Encoding and evaluation

| variable | read by | values and default | category |
|---|---|---|---|
| `LLVQ_MODEL` | `smoke`, `seal`, `ppl`, `mmlu`, `int4swap`, and 9 diagnostics | HF repo or local directory; `seal`, `smoke` and seven others default to `Qwen/Qwen3-0.6B`; `mmlu` has no default | tool |
| `LLVQ_CALIB` | `smoke`, `hcapture`, `hratio` | `wikitext2` (default), `c4`, `dclm-edu`, `wikitext2-test` | tool |
| `LLVQ_CALIB_SEED` | `smoke`, `hcapture` | integer; unset = the published draw | tool |
| `LLVQ_DAMPING` | `smoke` | GPTQ damping; unset = published | tool |
| `LLVQ_H_SHRINK` | `smoke`, `hcapture` | Hessian shrink ρ; unset = published | tool |
| `LLVQ_GAIN_SCALE` | `smoke` | gain multiplier; unset = published | tool |
| `LLVQ_SEQ_BLOCK` | `smoke` | unset or `1`; any other value refused | tool |
| `LLVQ_THREADS` | `smoke`, `errmap`, `f1recdump`, `tetra_cpu::pool` | thread count; about ncpu-4 on a shared machine; the CPU backend uses every core when unset | tool |
| `LLVQ_ARTIFACT` | `smoke` | path of the artifact to write; unset = none | tool |
| `LLVQ_RESUME` | `smoke` | shard to resume from; requires `LLVQ_ARTIFACT` | tool |
| `LLVQ_INT4_TYPES` | `smoke` | projection types written int4 g128, comma-separated | tool |
| `LLVQ_RESTORE_F16`, `LLVQ_RESTORE_Q4` | `sealed.rs` (`mmlu`, `ppl`) | projection types or `all`, taken from the checkpoint; both set refused | tool |
| `LLVQ_DTYPE` | `eval.rs` | `f32` (`ppl` default), `f16` (`mmlu` default), `bf16`, and their aliases | tool |
| `LLVQ_MMLU_ALLOC` | `mmlu` | `flat` (default), `proportional`, `proportional=<total>` | tool |
| `LLVQ_MMLU_DUMP` | `mmlu` | path of the per-question dump that `mmlupair` reads | tool |
| `LLVQ_GSM8K_DUMP` | `gsm8k` | path, one JSON line a problem | tool |
| `LLVQ_GSM8K_MAX_NEW` | `gsm8k` | token cap, default 1024 | tool |
| `LLVQ_GSM8K_THINK` | `gsm8k` | `0` (default), `1` | tool |
| `LLVQ_DATASET_REV` | `corpus.rs` | dataset revision pinned for the eval corpora | tool |
| `LLVQ_QTIP_DIR` | `planesbench` | upstream QTIP kernel directory (GPL v3, not redistributed) | tool |
| `LLVQ_CHAT_TEMP`, `LLVQ_CHAT_TOP_P`, `LLVQ_CHAT_SEED`, `LLVQ_CHAT_MAX_NEW`, `LLVQ_CHAT_THINK`, `LLVQ_CHAT_IDS`, `LLVQ_CHAT_SYSTEM` | `chat` | sampling, length, reasoning block, token-id echo, system prompt | tool |
| `LLVQ_ERRMAP_ARTIFACT`, `LLVQ_ERRMAP_BASELINES`, `LLVQ_ERRMAP_BASELINE_TRUST`, `LLVQ_ERRMAP_BLOCKS`, `LLVQ_ERRMAP_COMPARE`, `LLVQ_ERRMAP_EPS`, `LLVQ_ERRMAP_LOAD`, `LLVQ_ERRMAP_ONESIDED`, `LLVQ_ERRMAP_OUT`, `LLVQ_ERRMAP_PLAN`, `LLVQ_ERRMAP_PROBE_WINDOWS`, `LLVQ_ERRMAP_SAVE`, `LLVQ_ERRMAP_SAVE_TRUST`, `LLVQ_ERRMAP_SPLIT_B`, `LLVQ_ERRMAP_TRUST`, `LLVQ_ERRMAP_TRUST_SWEEP`, `LLVQ_ERRMAP_TYPES` | `errmap` | the error-map study's knobs; no job sets them | tool |
| `LLVQ_KL_ARTIFACT`, `LLVQ_KL_VAL`, `LLVQ_KL_SKIP` | `kltemp` | artifact, validation set, skipped blocks | tool |
| `LLVQ_ROT`, `LLVQ_H_DENSE` | `hcapture` | rotation on, dense Hessian | tool |
| `LLVQ_ALT_K` | `tetraalt` | alternative count | tool |
| `LLVQ_Q4B` | `radixstudy` | int4 bit study switch | tool |
| `LLVQ_STEPPED`, `LLVQ_RUN_DUMP`, `LLVQ_VERIFY_CACHE` | `run` (dense demo) | stepped decode, dump path, cache check (set means on) | tool |

## Measurement modes

| variable | read by | what it does |
|---|---|---|
| `LLVQ_CONTEXTS` | `fusedrun` | prompt lengths: time to first token and decode speed after each; allowed beside a config |
| `LLVQ_PREFILL_TOKENS` | `fusedrun` | prompt length for the prefill timing; allowed beside a config |
| `LLVQ_KV_PREALLOC` | `kvq.rs` | preallocated cache window |
| `LLVQ_GRAPH_AB`, `LLVQ_GRAPH_DIAG` | `fusedrun` | CUDA Graphs A/B and its diagnostic (lead closed) |
| `LLVQ_KV_AB`, `LLVQ_FUSE_AB`, `LLVQ_TRIM_AB` | `fusedrun` | cache store, fusion and trim A/Bs |
| `LLVQ_TIME_PHASES` | `fusedrun`, `model.rs` | per-phase timing of a token |
| `LLVQ_GQA_AB`, `LLVQ_GQA_PPL` | `fusedrun` | grouped attention timing, and its paired perplexity |
| `LLVQ_INT4_WIDE_CHECK` | `fusedrun` | int4 kernel against the host at widths above 14B |
| `LLVQ_GESTURE_AB`, `LLVQ_BOUNDARY_AB` | `fusedrun` | decode-step gesture and boundary A/Bs |
| `LLVQ_PERS_SEGMENT`, `LLVQ_PERS_4E` | `fusedrun` | persistent MLP segment, and 4E's row forms against the eager path |
| `LLVQ_BATCH_REF` | `batchrun` | the door's token fingerprint every batched row must reproduce |
| `LLVQ_BENCH_ARMS`, `LLVQ_SEG_ARMS`, `LLVQ_SEG_TETRA`, `LLVQ_TIME_EVENTS` | `planesbench` | arm selection, segment arms, Tetra segment, event timing |
| `LLVQ_TETRA_OCC`, `LLVQ_PERS_MICRO`, `LLVQ_TROWS`, `LLVQ_PERS_ARMS`, `LLVQ_DUMP_CUBIN` | `planesbench` (`persbench.rs`) | persistent Tetra arms, microbenchmarks, row loops, cubin dump |

## Tests

| variable | read by | what it does |
|---|---|---|
| `LLVQ_SEALED_ARTIFACT` | `llvq-artifact` tests | where to find the sealed file |
| `LLVQ_HF_FIXTURE`, `LLVQ_HF_MINI_FIXTURE` | `llvq-llm/tests/hfpack.rs` | rewrite the packed fixtures |
| `LLVQ_Q4_WIDE_FIXTURE_WRITE` | `llvq-llm/tests/proj_q4_wide.rs` | rewrite the wide int4 fixture |
| `LLVQ_ROT_FIXTURE_WRITE` | `llvq-cuda/tests/rotation_matches_rust.rs` | rewrite the rotation fixture |
| `LLVQ_TOKENIZER` | `llvq-runtime/examples/overhead.rs` | tokenizer file when no path is given |
| `LLVQ_UPDATE_OPENAPI` | `llvq-runtime/tests/openapi.rs` | rewrite the committed OpenAPI document |

## Python

`llvq-tetra` reads `LLVQ_HF_FUSED` (`0` or `1`), `LLVQ_HF_DEVICE` and `LLVQ_HF_TILE` (default 64, no range check).
`ops/mlx_tetra` reads `LLVQ_MLX_ABLATE`, and its door refuses every other `LLVQ_*` name. `ops/awq_speed.py` reads
`LLVQ_IMAGE_TAG` and `LLVQ_IMAGE_DIGEST`. The registry test does not read Python.

## Not environment variables

`LLVQ_GOLAY_CUH` and `LLVQ_PLANES12_CUH` are include guards that tests check, and `LLVQ_F1R_V3_NO_SIGN_PRMT` is a `-D`
macro. They are listed here because they are quoted the same way in the Rust code.
