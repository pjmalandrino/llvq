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
| 1 | `LlvqQuantizer` registered with `register_quantizer`: swaps `nn.Linear` for a `TetraLinear` that dequantizes in PyTorch, then runs a dense matmul. Rotation, int4 projections and int4 embeddings included | passed 2026-09-30: 253 weight digests against `decode_matrix`, and 64 greedy ids per prompt identical to `bin/run` | 0 $, Mac, CPU |
| 2 | Tetra decode as a torch op on Metal, from the existing shader | passed 2026-09-30: every one of the 118,665,216 blocks of the 4B decodes to the same point as the numpy path | 0 $, Mac |
| 2 bis | the fused matvec as a torch op, after the int4 staging fix | equality against a host reference in the kernel's own order, and the same tokens | 0 $, Mac, not started |
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
  Read again on 2026-09-30: the clause holds for the decode, which is integer and was gated exactly. It could not
  apply to the f64 chain, which Metal cannot run at all, and that is why the stage was narrowed rather than the
  tolerance widened.

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

Stage 1 passed on 2026-09-30, both gates (*measured*, `docs/mesures/hf-quantizer-4b-2026-09-30.txt`,
prereg `proofs/preregistration-hf-quantizer-2026-09-30.md`). 253 of 253 dequantized weight digests
identical to `llvq_artifact::decode_matrix`, and 256 of 256 greedy token ids identical to `bin/run`
over the four prompts, f32 on the CPU on both sides. `from_pretrained` reports no missing and no
unexpected key. The Python side is `llvq-hf/`.

Its kill criterion could not fire, and the prereg says so rather than claiming a pass: folding the
un-rotation into the dequantization removes the rotation from the forward pass, so no hook has to
host it. The real question, a kernel that reads rotated weights and rotates the activation, belongs
to stage 2 on Metal and stage 4 on CUDA.

Stage 2 passed on 2026-09-30 (*measured*, `docs/mesures/hf-metal-decode-4b-2026-09-30.txt`, prereg
`proofs/preregistration-hf-metal-decode-2026-09-30.md`). The served shader decodes the 4B under
PyTorch: 118,665,216 blocks, 2,847,965,184 coordinates, every point identical to the numpy decode,
in 38.4 s. No MSL was written, the shader's own `tetra48_probe` entry point is the op.

Its gate is not the one the plan wrote, and the prereg says why instead of widening it. Stage 1's
dequantization is an f64 chain by the format's design and **Metal has no f64**, which
`llvq-llm/kernels/llvq_rot.metal` already records by computing `1/sqrt(m)` on the host. So no
correct Metal implementation could match stage 1 bit for bit. What stays exactly gateable is the
integer part, the decode, and that is what stage 2 became. The operator took that decision on
2026-09-30.

Stage 2 bis carries what was cut: the fused matvec. It is blocked on a wall of its own, which is
written here because nothing else in the living documents carried it. `MetalRuntime::upload_int4`
stages `d_in · 4` bytes against a 32 KB threadgroup limit, so the Metal fused path refuses all three
sealed files at load, their int4 `down_proj` being `d_in` 9,728 at the 4B, 12,288 at the 8B and
17,408 at the 14B. The fix, tiling the staging by slices of 256 columns so each lane's accumulation
order is preserved, is designed and not written.

Stages 3 to 6 have not started. Each needs its own go.

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

- The go on stage 3, the Kernel Hub packaging, which needs the op of stage 2 and nothing more.
- The go on stage 2 bis, which needs the int4 staging fix first.
- Whether publishing the three sealed files (`docs/ETAT.md` §5) waits for stage 5.
