# Plan: serving Tetra files in vLLM

Goal: serve the sealed Qwen3-4B Tetra file in vLLM 0.26.0, as an out-of-tree quantization method, compressed on the
card. Its speed can then be compared with AWQ inside one engine. A prototype exists in `llvq-vllm/`, with four torch
ops in `llvq-tetra`. Nothing has run on a card. No speed number comes out of this document (rule 5).

## Why

Our engine loses to vLLM end to end, and the gap is the engine. On one L40S, vLLM decodes the 4B at 83.1 tok/s in FP16
and 200.5 with AWQ (*measured*, [awq-vllm-4b](mesures/awq-vllm-4b-2026-08-17.txt)). Our engine's dense f16 arm decodes at
43.4 and the served Tetra file at 133.3 (*measured*, `served-grouped-2026-11-02` on branch `debit-trims`). Its first
token after 8,192 tokens of prompt takes 35.0 s, because the prefill reuses the decode kernels four rows at a time.

vLLM prefills with GEMMs, batches, and carries speculative decoding and prefix caching. Inside it, Tetra against AWQ
is a comparison within one stack, which rule 5 allows. The `transformers` path (`llvq-tetra` 0.1.0) gives no speed by
design.

## The API at the pinned version

Read on 2026-11-02 in the sources of tag `v0.26.0` (commit `568afb3a`), the version of the pinned image
`vllm/vllm-openai:v0.26.0` (digest `ffb2d59b`). Every hook below is public and needs no patch of vLLM.

| hook | where | what the plugin does |
|---|---|---|
| entry point `vllm.general_plugins` | `plugins/__init__.py`; loaded by `engine/arg_utils.py:769`, `v1/engine/core.py:112`, `v1/worker/worker_base.py:247` | `llvq = llvq_vllm:register`, so every process registers the method before it reads a config |
| `register_quantization_config` | `layers/quantization/__init__.py` | registers `llvq`, the `quant_method` that `hfpack` writes |
| `QuantizationConfig.from_config` | `model_loader/weight_utils.py:get_quant_config` | reads the record table from `config.json`, checks the Tetra fingerprint |
| `get_quant_method(layer, prefix)` | `linear.py:274`, `vocab_parallel_embedding.py:278`, `attention.py:611` | linear layers and the int4 vocabulary get our methods; attention gets `None`, so the KV cache stays f16 |
| `packed_modules_mapping` | `model_loader/utils.py:configure_quant_config` | names the records behind `qkv_proj` and `gate_up_proj`, set before any layer is built |
| `create_weights`, `process_weights_after_loading`, `apply`, `embedding` | `base_config.py`, `linear.py` | parameters named as the file names its tensors; the served layout built on the device after the load |
| `get_cache_scale_mapper()` | `models/utils.py:414` | a `WeightsMapper` applied to the whole weight stream; it carries the rotation tables to their owner layers |

vLLM checks for missing weights only when a model is not quantized (`DefaultModelLoader`). So every parameter of ours
records the shards its loader received, and a layer with a hole is refused at load, by name.

## The image, and how the plugin installs

The image ships torch 2.11.0, CUDA 13.0.2 and Python 3.12, requires transformers 5.5.3 or later, and installs `nvcc`
for runtime compilation (`docker/Dockerfile`, `requirements/cuda.txt`). `llvq-tetra` declares torch 2.14 or later for
its `transformers` path. A resolver would replace the image's torch and break every compiled kernel of vLLM.

So the plugin is a sibling package, `llvq-vllm/`, with no declared dependency, installed by
`pip install --no-deps ./llvq-tetra ./llvq-vllm`. It reuses the reader, the decode tables and the CUDA extension of
`llvq-tetra`, so nothing is derived twice. `import llvq_tetra` also imports its `transformers` quantizer. It succeeds on
transformers 5.5.3, vLLM's floor, and on 5.22.1 (*measured*, Mac). The image's exact version is read in the job.

The tests run on torch 2.15.0 and again on torch 2.11.0, the image's. The older Dynamo found one defect the newer one
let pass: a buffer name formatted from an int inside traced code, which it treats as symbolic. The names are now built
at load.

## The five design questions

### 1. One rotation for q, k and v, and for gate and up

vLLM fuses q, k and v into one `QKVParallelLinear`, and gate and up into one `MergedColumnParallelLinear`. One `apply`
call sees every shard, so the plugin rotates once per rotation key. That is `LLVQ_ROT_SHARE=1` of the served config,
with no cache keyed on the activation, which the `transformers` path needs because it calls q, k and v one by one.

On the 4B the shards of one layer are not of one kind (*computed*, from the record table):

| vLLM layer | shards | rotation |
|---|---|---|
| `qkv_proj` | q and k Tetra, v int4 g128 | q and k share key `…0110feed`; v reads the activation unrotated |
| `gate_up_proj` | gate and up Tetra | one key, `…0112feed` |
| `o_proj` | int4 g128 | none |
| `down_proj` | Tetra in layers 0 to 11 and 24 to 35, int4 in 12 to 23 | `…0113feed`, odd factor 19 |

That is 96 rotations a token, the count the served engine prints. The outputs are concatenated in vLLM's column order.

The file carries each rotation once, as `llvq.rotations.<key>.signs` and `.small`, and no layer owns those names. Each
key is given an owner, the vLLM layer of its first record in sorted order. The owner holds the two tables as parameters,
and the mapper renames the file's tensors onto them. A layer that uses a key it does not own reads the owner's tables
after the load. A table that no record uses is dropped by name.

### 2. The int4 records and the tied head, never widened

The 84 int4 records hold 0.409 GB served, and 1.541 GB widened to f16. The tied table holds 0.219 GB in int4 g64, and
0.778 GB in f16 (*computed*, `llvq-vllm/tests/fixtures/qwen3-4b-config.json`). Widening both would add 1.69 GB to a
model that holds 1.378 GB in all, the 1.38 GB the engine prints on the card.

The int4 records go to `tv_q4_h`, bound as `llvq_cuda::tv_q4`. Qwen3-4B ties its head by making `lm_head` the same
module as `embed_tokens` (`models/qwen3.py`). So one method serves both: `embedding` gathers rows in torch, exactly as
`emb_q4_gather` narrows them, and `apply` computes the logits through `tv_emb_q4_h`, bound as `llvq_cuda::tv_emb_q4`.
The vocabulary, 151,936 rows, is a multiple of vLLM's padding of 64, so no row is added.

### 3. Prompts with M > 1

| tokens in the call | Tetra | int4 g128 | tied head |
|---|---|---|---|
| up to `LLVQ_VLLM_ROWS_MAX`, 16 by default | `tv_tetra48`, once a row | `tv_q4`, once a row | `tv_emb_q4`, once a row |
| above | dequantize to f16, then a cuBLAS GEMM | the same | the same, 16,384 vocabulary rows at a time |

The row path runs the served kernels and keeps our engine's arithmetic for those projections. The dense path rebuilds
`W'` in the rotated basis and multiplies the rotated activation, one shard at a time. Its scratch is 49.8 MB at most,
83.9 MB for a head chunk (*computed*). It rounds weights and activations to f16, as vLLM's FP16 arm does, so it is held
to a tolerance and never to identity.

The dequantization in the prototype is written in torch: the decode by table gathers, then the two scales. It runs on a
CPU in the tests and on the card unchanged. On the L40S it should take 0.15 to 0.35 s for the 2.86 billion Tetra weights
of one pass (*estimated*, from 1 to 2.5 kB of memory traffic a block). A CUDA dequantization kernel is the next op: one
thread a block, built from the served `f1r_load` and `tetra48_decode_f`, with no new arithmetic. Until it exists, a time
to first token measures the torch decode as much as the format. On a card under 70 GiB, vLLM 0.26.0's offline `LLM`
class prefills 8,192 tokens in one chunk (`engine/arg_utils.py:2472`), so an 8k prompt is one dequantization.

### 4. What CUDA graphs and `torch.compile` need from the ops

vLLM 0.26.0 compiles the model with Dynamo and Inductor, the token count dynamic. It captures full CUDA graphs for decode
batches and piecewise graphs for prefills (`FULL_AND_PIECEWISE`). The sizes are 1, 2, 4, then steps of 8 to 256 and of
16 above, with no warmup run before a capture (`config/compilation.py`). The AWQ figure of 200.5 tok/s was measured
under these defaults.

Three properties follow, and the prototype holds each one:

| need | how | checked |
|---|---|---|
| no branch on the token count in traced code | each op that chooses a path by M is one opaque op, `llvq_vllm::*`, with a fake implementation | `opcheck` on every op; a full-graph compile that does not recompile from M = 3 to M = 9 |
| nothing that synchronizes with the host | the ops allocate and launch; no `.item()`, no copy to the host | by reading; only a card can confirm |
| no first-use copy inside a capture | the extension, the decode tables and `invnorm` move to the card at load, in `process_weights_after_loading` | `prepare` cached before any forward |

Rule 3 forbids CUDA graphs in our core. The core is the Rust engine, and these graphs are vLLM's. Whether the rule
covers them is the operator's decision, below.

### 5. Tolerance, not identity

Bit identity with our engine cannot hold in vLLM. Its attention, norms and rotary embedding are other kernels, and the
dense path rounds to f16. So the gate is the tolerance the grouped attention passed on 2026-11-01: the paired
perplexity ratio within ±0.12 % on 12 WikiText-2 windows, plus the 256-token reference line. The draft prereg is
[`proofs/BROUILLON-preregistration-vllm-tetra.md`](../proofs/BROUILLON-preregistration-vllm-tetra.md).

## What exists

| piece | file | held by |
|---|---|---|
| four CUDA ops: `tv_tetra48`, `tv_q4`, `tv_emb_q4`, `rot_apply_rows` | `llvq-tetra/llvq_tetra/csrc/tetra_cuda.cu` | `test_cuda_sources.py`: schemas against callers, digests of the shipped sources |
| ten shipped CUDA sources, four of them new | `llvq-llm/src/hfpack.rs`, `bin/tetratables` | `the_shipped_cuda_closure_is_complete`, `the_cuda_binding_includes_the_served_list` |
| record plan, served buffers, ops, layer, loaders | `llvq-vllm/llvq_vllm/` | 83 tests on the CPU references, against the f64 reconstruction |
| the vLLM adapter | `llvq-vllm/llvq_vllm/vllm_glue.py` | imports only where vLLM is installed; never run |

On the fixture, every layer and both paths read a worst row error of 2.0e-4 to 4.4e-4 against the f64 dense product,
under a bar of 1e-2 (*measured*, Mac CPU). The CUDA stage measured 3.4e-4 to 3.7e-4 on the card for the same check.
Eleven mutants of the new code were each killed: rotation skipped, shards reordered, gain bit misplaced, tail dropped,
hole accepted, wrong owner, nibbles swapped, signs inverted, fake shape wrong, schema drifted, includes reordered.

## What has not run

Everything on a card: the four ops never compiled, and the adapter never ran inside vLLM. The 4B's own mixed `qkv_proj`
is tested on the fixture's q and k with a synthetic int4 v, because no fixture carries that layout. The 8B, the 14B and
tensor parallelism are refused by name.

## Open decisions

- **CUDA graphs under rule 3.** Run Tetra in vLLM with its default graphs, as the AWQ figure was, or run both arms
  eager. The draft prereg measures both and publishes the first.
- **Tolerance instead of identity** for this path, as decision T8 allowed the grouped attention.
- **The dequantization before a time to first token**: measure with the torch decode now, labelled, or write the CUDA
  kernel first.
- **The references of our engine**: the per-window perplexities and the 256-token line exist in two journals of
  2026-11-01 and 11-02 on branch `debit-trims`. One image cannot hold both engines, so re-measuring them is a second job.
- **The package layout.** `llvq-vllm` imports `llvq_tetra`, whose import pulls the `transformers` quantizer. It works
  from transformers 5.5.3 on; splitting that import would be a change to the published package.
- **A spending cap.** None is in force (`ETAT.md` §5), and one is owed before this job.

## Next, each on its own go

1. A probe in the pinned image on `cpu-upgrade`, about $0.01 (*estimated*): install, register, read the config, and
   check by introspection the hooks of the table above. It costs a minute and catches an import failure.
2. The card job of the draft prereg, about $0.55 (*estimated*).
3. The CUDA dequantization kernel, then the four-row prefill kernel `tv_tetra48_rows_h` for 2 to 16 tokens.
