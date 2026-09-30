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
| M1 | the fused Tetra matvec as a torch op, the weights resident compressed, the rotation in the forward pass | passed 2026-09-30: 256 ids of 256, the 168 Tetra projections resident in 0.749 GB, loaded in 6.5 s | 0 $, Mac |
| M2 | the int4 staging fix, then the int4 matvec as a torch op | passed 2026-09-30: bit-identical to the served kernel at all three tiles, 256 ids of 256, all 252 projections resident, 2.750 GB allocated | 0 $, Mac |
| M3 | the quantized embedding resident, through `emb_q4_gather_metal` | the same ids | 0 $, Mac |
| 3 | Kernel Hub packaging for Metal (`kernel-builder`), loaded with `get_kernel` | `kernel-abi-check` green; stage 2's tokens reproduced from the Hub-loaded kernel | 0 $, Mac |
| 4 | CUDA: the NVRTC source becomes a precompiled torch extension | the per-row check against the dense reference, then the same ids as `bin/run` on the same card | launched 2026-09-30, **$0.18 spent on three two-minute build failures**, running on an `l4x1` at ~$0.25 estimated under a $0.53 cap |
| 5 | Pip package, model card, 4B file pushed to the Hub in the new layout | a clean environment runs `pip install` then `from_pretrained` and reproduces the tokens | 0 $ |
| 6 | Upstream PR to `transformers` | accepted or refused by the maintainers; not ours to decide | 0 $ |

## The CUDA test job, launched

One job, because the registry shows 30 to 70 minutes of queue per job and the run itself is minutes:
batching the whole question into one launch is what the queue makes rational.

**What it would answer.** Whether the served CUDA kernel can be reached from `torch` as a
precompiled extension and give the same tokens. Three unknowns, all of them local to the card: that
`nvcc` compiling `llvq-cuda/kernels/llvq_tetra48.cuh` outside NVRTC keeps the arithmetic, that the
buffer binding holds, and that the int4 and `rot_apply` arms work there. CUDA's wall is not Metal's:
`tv_q4_h.cu` serves the 4B's `down_proj` at `d_in` 9,728 today, so M2's blocker does not exist on
that side.

**The arms, in order, so a failure is diagnosable.**

1. Build the extension with `nvcc` in the job. A build failure costs two minutes and answers the
   question on its own.
2. The per-row check on one matrix of each shape, against the dense reconstruction, exactly as the
   Metal arm was checked before any token was generated.
3. 64 greedy ids on the four prompts of `bin/run`, and `bin/run` itself on the same file on the same
   card in the same job, so the reference is local and no number crosses a machine.
4. The resident memory, measured.

**What has to exist first**, and none of it needs a card:

- `hfpack` in the CUDA image: the `cargo build --bin` list of `ops/Dockerfile.cuda`, its runtime
  `COPY`, and `UPLOAD_ALLOW` in `ops/run.py`. That third list is the one that has bitten four times.
- `llvq-hf/` and `llvq-cuda/kernels/` in `UPLOAD_ALLOW`.
- `torch` and `transformers` in the job, which the image does not carry. The launcher pip installs
  them, about three minutes and $0.09.
- A CUDA binding beside the Metal one, `llvq-hf/llvqhf/csrc/tetra_cuda.cu`, which is the same shape
  of file: no arithmetic, one `#include` of the served header.

**The estimate.** l40sx1 at $1.80 an hour. Build five minutes, `hfpack` two, pip three, the load and
the four prompts five: **15 minutes billed, $0.45**. The cap is **$1.20**, which is 40 minutes, and
the job is killed rather than allowed past it. For scale, the registry's last six jobs on this flavor
billed 5 to 157 minutes for $0.16 to $4.71.

**What it cost, 2026-09-30.** Four launches on `l40sx1`, **$0.18**, and arm 1 never passed. Each
failure took two minutes and named its cause, which is what arm 1 is for: `nvcc` refusing
`-std=c++17` where torch's headers demand C++20, then the include order leaving `F1rTables`
undefined, then `TETRA48_ORDER` undefined in device code because NVRTC compiles a whole unit as
device code and `nvcc` does not. Two of the three are now held by tests on the Mac at 0 $, and the
third was a fact already in hand from the Metal path. The fourth launch then sat in the queue **3 h
47** with no card and was canceled at $0 billed.

**The card is an `l4x1` from 2026-09-30.** Same architecture, sm_89 for both, so the extension builds
the same code; 24 GB against a 2.8 GB object; a pool that is not empty; and $0.80 an hour, so the
estimate falls to about **$0.25** under a **$0.53** cap. `run.py bench` refuses any card outside
`BENCH_FLAVORS` and the launch carries `--any-flavor`, whose duty is to name the card in every
figure. The four arms are a build, a per-row identity, 64 tokens and a byte count, so not one of them
is a throughput and rule 5 has nothing to divide. Written in
`proofs/preregistration-hf-cuda-2026-09-30-ECARTS.md`.

**What it cannot answer.** Nothing about throughput: the per-token dispatch of this path is not the
served engine's, and no tok/s from it is divided against anything (rule 5). Nothing about the 8B and
the 14B. Nothing about a second kind of card.

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

## What Metal needs, and what each lot buys

Stages M1 to M3 replace what the plan called stage 2 bis. The compression is on **disk** today: a
loaded model is dense, 8.05 GB at f16 on the 4B against 1.42 GB of file. What makes it exist in
memory is the fused matvec, which multiplies without ever writing the matrix. Every shader it needs
is already in this repository, `tv_tetra48_metal`, `tv_q4_metal`, `rot_apply_metal` and
`emb_q4_gather_metal`; what is missing is the wiring and one fix.

Four facts decide the shape of the work, all read in the code on 2026-09-30.

- **The int4 matvec is the blocker.** 84 of the 4B's 252 records are `Int4G128`, 21 % of the weights.
  `tv_q4_metal` stages the whole activation with no tile, `d_in · 4` bytes against a 32 KB
  threadgroup limit, so it refuses `down_proj` at `d_in` 9,728. The fix is to tile the staging by
  slices of 256 columns, which preserves each lane's accumulation order and therefore the
  arithmetic. Nothing in the living documents carried this wall before.
- **The rotation moves into the forward pass.** The weights stay in the rotated basis, so the
  activation must be rotated before each matvec. Stage 1 dissolved that question by un-rotating at
  load; that is no longer available, since un-rotating means materializing.
- **The kernel is not bit-identical to the dense path, by construction.** It reads the tail as
  `half` and the row scales as `float`, where the file stores f32 and f64. Its gate is therefore the
  one the repository already uses for it: per-row against f64 within a tolerance, and identical
  tokens.
- **There is no batch.** `tv_tetra48_metal` takes one activation vector, and `transformers` calls
  `forward` with `[B, T, d_in]`. A prefill of T tokens is T dispatches per projection.

Resident weight memory on the 4B, *computed*:

| arm | resident |
|---|---|
| dense f16, what stage 1 gives | 8.05 GB |
| M1, the 79 % Tetra compressed, int4 and embedding dense | 3.1 GB |
| M1 + M2 | 1.93 GB |
| M1 + M2 + M3 | 1.37 GB |

M1 is indivisible: the rotation, the matvec and the residency have no meaning apart, since removing
the materialization is the whole point.

**M1 passed on 2026-09-30** (*measured*, `docs/mesures/hf-metal-m1-4b-2026-09-30.txt`, prereg
`proofs/preregistration-hf-metal-m1-2026-09-30.md`). The 4B answers with its Tetra weights never
materialized: 256 greedy ids of 256 against `bin/run`, the 168 Tetra projections resident in 0.749 GB
against 11.4 dense, 5.436 GB allocated on the device against 16.1 computed dense, and the load down
from 153.7 s to 6.5 because nothing is dequantized any more.

**What M1 found is bigger than M1.** Dropping the tail entirely costs 8.79 % of the output on
`k_proj` and 3.53 % on `down_proj`, measured per row. The token gate misses that defect over eight
ids on all four prompts and over all 64 on two of them; it catches it at token 13 on the best one.
Stages 1, 2 and M1 all took token identity as their gate, and this is the first measurement of what
that gate cannot see. The per-row check against the dense reconstruction finds the same defect in 20
seconds with a 3,400-fold margin.

**M2 passed on 2026-09-30** (*measured*, `docs/mesures/hf-metal-m2-4b-2026-09-30.txt`, prereg
`proofs/preregistration-hf-metal-m2-2026-09-30.md`). The 4B now holds **no projection dense**. The
new `tv_q4_metal_tiled` sits beside the served `tv_q4_metal` in the same file, 81 insertions and 0
deletions, and gives the same f32 value for value at tiles 256, 2,048 and 8,192 wherever the served
kernel runs. The model gives the same 256 ids of 256, and the device allocation falls from M1's 5.436
GB to **2.750**, against a signed interval of [2.6 ; 3.1] and a point estimate of 2.75. The 1.558 GB
of parameters left is the f32 embedding and nothing else.

The four signed predictions all held, the first clean sheet on this branch. The measured table, where
16.1 GB is *computed* because the dense f32 arm does not fit:

| | dense f32 | M1, Tetra | M2, Tetra + int4 |
|---|---|---|---|
| ids vs `bin/run` | reference | 256/256 | 256/256 |
| mps allocated | 16.1 computed | 5.436 GB | **2.750 GB** |
| parameters and buffers | | 4.641 GB | 1.558 GB |
| projections resident | | 0.749 GB for 168 | **1.158 GB for all 252** |
| load | 153.7 s | 6.5 s | 7.7 s |

**No speed is claimed at any stage.** The per-prompt wall times of the M1 and M2 runs cross each
other in both directions, which is machine contention and not a kernel.

One control nearly went void, and the assertion is what caught it: `tv_q4_metal_tiled` comes first in
the shader and the served function after it, so a mutation "in the tiled kernel" obtained by
partitioning on its name lands on both, the reference moves with the subject, and all three mutants
read as survived. `llvqhf/checkq4guards.py` now closes the region at the next entry point.

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
- **Whether the per-row check becomes a gate**, on one matrix of each shape, with token identity kept
  beside it. M1 measured that four prompts and 64 greedy ids cannot see a 3 to 9 % per-row error,
  while the per-row check sees it in 20 s with a 3,400-fold margin. This changes what a gate is in
  this plan, so it is the operator's.
- The go on M3, the quantized embedding, which is the last 1.558 GB and takes 2.750 GB to about 1.4.
- Whether publishing the three sealed files (`docs/ETAT.md` §5) waits for stage 5.
