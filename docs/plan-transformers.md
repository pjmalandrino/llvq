# Plan: loading Tetra files in `transformers`

Goal: a sealed Tetra file loads with `from_pretrained` and generates the same tokens as our engine. Nothing is run
until the operator gives a go on each stage (rule 1). Stages 0 to 3 cost 0 $ and run on the Mac.

## Why

Nothing but our engine reads a `.llvq` today (`docs/hf-model-card.md`). `bin/export` bridges to `transformers`
by writing full f16, about 8 GB at 4B: an interchange artifact, never a distribution format. This plan targets
loading the **compressed** file.

## Facts the plan rests on

Verified on 2026-09-28 in the upstream sources, `main` branch:

- `transformers/quantizers/auto.py` exposes `register_quantizer` and `register_quantization_config`. A method can be
  plugged in without a PR.
- In-tree integration is a PR reviewed by the `transformers` maintainers. Their guide requires a pip-installable
  package, ideally with precompiled kernels.
- Vector methods are already in tree: `aqlm`, `vptq`, `higgs`, `spqr`.
- The Kernel Hub (`huggingface/kernels`) requires kernels registered as `torch.ops.<namespace>`. Backends include
  `cuda` and `metal`. On macOS it requires Metal 4.0 and a deployment target of 26.0.

What we have: the CUDA kernel `llvq-cuda/kernels/llvq_tetra48.cuh`, compiled by NVRTC at launch, the Metal shader
`llvq-llm/kernels/llvq_tetra48.metal`, and a Python package with torch adapters (`ops/llvqtune`).

## Stages

| stage | content | gate | cost |
|---|---|---|---|
| 0 | Map the sealed file to safetensors: codes, gains, row scales, f32 tail, int4 records, rotation seeds, plus `config.json` with a `quantization_config` block | a Python reader rebuilds every tensor bit for bit against `llvq-artifact` on the 4B | 0 $, Mac |
| 1 | `LlvqQuantizer` registered with `register_quantizer`: swaps `nn.Linear` for a `TetraLinear` that dequantizes in PyTorch, then runs a dense matmul. Rotation, int4 projections and int4 embeddings included | `from_pretrained` loads the 4B. 64 greedy tokens identical to `bin/run` on the dense reconstruction, same prompt | 0 $, Mac, CPU or MPS |
| 2 | Tetra decode as a torch op on Metal, from the existing shader | the op matches stage 1's dequant bit for bit on every 4B matrix; same 64 tokens | 0 $, Mac |
| 3 | Kernel Hub packaging for Metal (`kernel-builder`), loaded with `get_kernel` | `kernel-abi-check` green; stage 2's tokens reproduced from the Hub-loaded kernel | 0 $, Mac |
| 4 | CUDA: the NVRTC source becomes a precompiled torch extension, built by `kernel-builder` | `oracle`-style check against the f64 rows; same tokens as `fusedrun` under `configs/qwen3-4b-tetra-e4.json` | small, one L40S job, to price before the go |
| 5 | Pip package, model card, 4B file pushed to the Hub in the new layout | a clean environment runs `pip install` then `from_pretrained` and reproduces the tokens | 0 $ |
| 6 | Upstream PR to `transformers` | accepted or refused by the maintainers; not ours to decide | 0 $ |

## Kill criteria

Written before stage 0, to be timestamped in the prereg:

- Stage 0 fails if a field of the sealed file has no faithful safetensors representation. Document it and stop.
- Stage 1 fails if the quantizer hook cannot host the per-group rotation without patching the model code. That
  would close the out-of-tree route; the in-tree PR becomes the only route, and it is reassessed then.
- Stage 2 fails if the Metal op does not match bit for bit. That is a defect to fix, not a tolerance to widen.

## What this plan does not claim

- No speed number. `transformers` is not a throughput engine, and any tok/s it gives is never divided against our
  engine or vLLM (rule 5).
- Batch 1 and the `rot_apply` wall above 14B carry over unchanged (`docs/format-noyau.md` §8).
- Acceptance of the stage 6 PR is not estimated.

## Where it stands

Stage 0 passed on 2026-09-28: 1602 fields of the served 4B rebuilt bit for bit by an independent reader, 1.422 GB of
directory against 1.418 of sealed file (*measured*, `docs/mesures/hf-safetensors-4b-2026-09-28.txt`, prereg
`proofs/preregistration-hf-safetensors-2026-09-28.md`). Its kill criterion did not fire. The three format decisions
were taken by the operator before the code and are recorded in that prereg §3: our own tensor naming, the disk's bytes
as the code payload, the rotation carried as its two tables.

Stages 1 to 6 have not started. Each needs its own go.

## Where the code lives

The Python side is developed here, in a top-level `llvq-hf/`, and extracted at stage 5. Decided by the operator on
2026-09-29.

Three facts settle it. `hfpack` reads a `.llvq` through `llvq-artifact` and `llvq-quant`, so it is a workspace crate's
binary and cannot move. A pip-shaped Python package inside this repository is already the practice, `ops/llvqtune`
carries its own `pyproject.toml`, `uv.lock` and tests. The lab rules are bound to this repository: preregs in `proofs/`
with their `.ots` anchors, journals in `docs/mesures/`, and every stage below carries a gate, so developing stages 1 to
4 elsewhere would separate the audit trail from the code it attests.

`llvq-hf/` is self-contained from the first commit: its own `pyproject.toml`, its own tests, and a fixture of a few
hundred kilobytes so no test needs the 1.4 GB object. Extraction is then `git subtree split -P llvq-hf`, which keeps the
history, and the wheel of stage 5 is published from the repository that comes out. What is published before that is the
kernels, which the Kernel Hub takes as Hub repositories, and they are outputs rather than homes.

A fork of `transformers` is not a home either. The in-tree guide of stage 6 requires the pip package to exist first, so
the fork is downstream of it. The repository's own precedent for an upstream contribution is
`docs/upstream/candle-broadcast-matmul/`, an issue, a patch and a reproduction, with no fork.

`ops/llvq_hf_check.py` holds the 48-bit unpacking that stage 1 needs. It moves into the package at stage 1 and the
script keeps its name as a thin caller, so the unpacking never exists twice. The provenance line of the stage 0 journal
is updated at the same time.

## Open decisions

- The go on stage 1.
- Whether `quantization_config` stays pretty printed at 131 KB, or the record table moves to a side file the quantizer
  reads. Measured at stage 0, decided at stage 1.
- Whether publishing the three sealed files (`docs/ETAT.md` §5) waits for stage 5.
