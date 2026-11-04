DRAFT, not stamped, not committed

# Preregistration. How fast a 16-lane CUDA-core GQA decode reads a Qwen3-4B KV cache on our L40S

**Draft written on 2026-11-02, to be committed and TIMESTAMPED BEFORE the run.** Operator go to launch: **not given**.
Cost: about **$0.15 to $0.42** on l40sx1, *estimated* below (§8), timeout 25 min, so $0.75 at worst. Cap asked:
**$1**, which covers one retry. Spent before it: $248.84 over 237 rows of the registry (*computed* from
`docs/data/jobs.csv` on 2026-11-02).

## 1. Why this run exists

The 4B's decode attention reads its cache at 0.25 to 0.26 of 864 GB/s at 2,048, 4,096 and 8,192 positions
(*computed* from `docs/mesures/served-grouped-2026-11-02.txt`). The attention lead projects its own split-KV kernel at
η 0.60 to 0.85, and no source gives an Ada figure for a GQA decode kernel at batch 1.

FlashInfer's CUDA-core decode is the closest kernel that exists. At f16 and head_dim 128 it gives each query head 16
lanes of 8 dims, the lead's own lane partition, and it splits the cache in chunks of at least 256 positions, the
lead's split size (`include/flashinfer/attention/decode.cuh`, `SingleDecodeWithKVCacheDispatched`, read at tag
v0.6.14 on 2026-11-02). This run measures its read rate on our card and our geometry, with a cold L2 and no host or
mask work. It sets the planning band of η, the mapping set of build step 2, an input to the 14B's choice of cores, and
a check on the int8 design. It informs the build plan and decides nothing about the lead (METHODE §1).

It is an anchor, and it differs from our kernel in four ways. Its exp2 is the MUFU `ex2.approx`, where ours is a
13-instruction polynomial. Its chunks are merged by a second kernel, `MergeStates`, where ours folds in the last CTA.
Each CTA holds the G heads on `threadIdx.y` and two position streams on `threadIdx.z`. Its stages are smaller: 8
positions of K and V at G = 4, by the shared-memory formula of `SingleDecodeWithKVCacheDispatched`.

## 2. What runs

One job, l40sx1, image `vllm/vllm-openai:v0.26.0`, the tag every vLLM job of the repository has used, manifest digest
`sha256:ffb2d59b…` recorded beside it (`docs/mesures/awq-vllm-4b-2026-08-17.txt`). The tag ships flashinfer 0.6.14 and
`cuda-nvcc` for FlashInfer's runtime JIT (`docker/Dockerfile` and `requirements/cuda.txt` at tag v0.26.0, read
2026-11-02). Script `ops/fi_decode_bench.py`, uploaded to the bucket and checked by sha256 in the job; job script
`ops/jobs/fi-decode-anchor.sh` (drafts beside this file, in `code/attention/`).

For each arm and each length L in {2,048, 4,096, 8,192}: 36 distinct per-layer KV sets of the 4B geometry (8 KV heads,
head_dim 128, layout HND `[8, L, 128]`, the Grow store's per-layer layout), and one query of 8G heads. One pass is 36
calls of `flashinfer.single_decode_with_kv_cache`, so it reads 36 × 2 × 8 × L × 128 elements and reuses no set:
0.30 GB at 2,048 in f16, above the 96 MB L2. The pass is captured once as a CUDA graph and timed with CUDA events
around each replay; a pass that cannot be captured is timed eagerly and flagged. 2 discarded and 7 timed rounds; inside
a round every (L, arm) pair runs in one fixed order. One profiler pass per (L, arm) gives the kernel-only sum and the
kernel names. Set 0 of each (L, arm) is checked against an f32 torch reference.

Arms. FlashInfer 0.6.14's CUDA-core decode accepts group sizes 1, 2, 3, 4 and 8 only (`DISPATCH_GQA_GROUP_SIZE`,
`include/flashinfer/utils.cuh` at v0.6.14), so the 14B's G = 5 is bracketed by 4 and 8 there. The tensor-core path
(`use_tensor_cores=True`, the FA2 prefill kernel) is tried at 5 directly.

| arm | cores | G | KV dtype |
|---|---|---|---|
| P (primary) | CUDA | 4 | f16 |
| C8 | CUDA | 8 | f16 |
| P8b | CUDA | 4 | e4m3 |
| C8-8b | CUDA | 8 | e4m3 |
| T4, T5, T8 | tensor | 4, 5, 8 | f16 |
| T4-8b | tensor | 4 | e4m3 |

Derived quantities, *computed* after the run:

- η(arm, L) = bytes of one pass / (t × 864 GB/s), on the event median, with the range from the slowest and fastest
  rounds;
- E8 = η(P, 8,192);
- F = η(P, 2,048) / η(P, 8,192), the fill sensitivity;
- H = η(C8, 8,192) / η(P, 8,192), the group-size sensitivity on CUDA cores;
- Q = R(P8b, 8,192) / R(P, 8,192), where R is the read rate in GB/s of unique bytes, the 8-bit proxy.

Ratios between arms are formed round by round, since all arms run in every round.

## 3. Gates, each one voids the run

1. `nvidia-smi` names an L40S.
2. The job prints `vllm 0.26.0` and `flashinfer 0.6.14`, and the bench script's sha256 matches the uploaded file.
3. Arm P is timed at all three lengths, and its set-0 error is under the bound (2e-2 of max |ref| in f16).
4. The JSON holds every arm, timed or marked void with its exception.

If arm P was timed eagerly, its readings use the profiler's kernel-only sum instead of the event median, declared
here before the run.

## 4. Outcomes reported, never gated

Every arm's medians, ranges and η at each length; kernel names and kernel-only sums; first-call times (JIT or cache);
the void arms and their errors; the tensor-core arms against the CUDA-core arms at equal G.

## 5. Signed predictions

| quantity | point | interval |
|---|---|---|
| η(P, 8,192) = E8 | **0.75** | [0.50, 0.92] |
| η(P, 4,096) | **0.65** | [0.40, 0.88] |
| η(P, 2,048) | **0.50** | [0.25, 0.80] |
| F | **0.67** | [0.40, 0.95] |
| H | **0.90** | [0.60, 1.05] |
| η(T4, 8,192) / E8 | **0.85** | [0.50, 1.10] |
| Q | **0.80** | [0.45, 1.00] |
| T5 timed (not void) | **yes** | |

Reasoning, *estimated*. FlashInfer reports near-full bandwidth for MHA decode on an RTX 6000 Ada, the same AD102 chip
(blog, local copy `wave3-verify/w7-flashinfer/blog.txt`). Batch 1 with 8 KV heads gives 256 CTAs of 128 threads at
8,192 positions, about 1.8 per SM, each with one to two 4 KB stages in flight, so latency hiding is thin. At 2,048
only 64 CTAs run on 142 SMs. Two kernels a call add a few microseconds a layer, which weigh most at 2,048, where a pass
reads 0.30 GB in about 0.35 ms at full rate. The known flaw: graph replay removes the CPU gaps that an eager path pays,
which the in-situ A/B (prereg-attention-2) measures instead.

## 6. Decision rules

Four readings, each evaluated alone, each changing the build plan and none pronouncing a kill (METHODE §1). A reading
whose arms are void takes its "otherwise" line.

**Reading 1, the planning band of η, from E8.**

- **E8 ≥ 0.70:** the band [0.60, 0.85], central 0.75, is kept, and the kernel-only bench (`prereg-attention-2a`)
  predicts with it.
- **0.45 ≤ E8 < 0.70:** the band becomes [0.45, 0.70]. Every 8k projection of attention.md §3.7 is recomputed before
  `prereg-attention-2a` is stamped.
- **E8 < 0.45:** the band becomes [0.30, 0.50], and the lead's 8k projection falls to ×1.10 to ×1.30 (*computed*:
  7.12 ms plus 1.208 GB at η 0.30 to 0.50, against 12.90 ms). Before build step 2, the operator is asked whether to
  write the CUDA kernel now or after the floors that do not depend on η. Build step 1 continues: it costs $0, and the
  speculation lead's exact verify needs it.
- **Otherwise** (a gate failed): not settled; raw output kept; the operator decides on a rerun.

**Reading 2, the mapping set of build step 2, from F.**

- **F ≥ 0.80:** step 2 writes the CTA-per-split mapping only.
- **F < 0.80:** step 2 also writes the CTA-per-unit mapping (four times the CTAs at 2,048, bit-neutral, attention.md
  §3.2), and the kernel-only bench times both.
- **Otherwise** (F not formed): both mappings are written.

**Reading 3, the 14B's cores, an input to V9, from H and the tensor-core arms.**

- **H ≥ 0.85:** the 14B's G = 5 template is planned on CUDA cores.
- **H < 0.85 and η(T8, 8,192) > η(C8, 8,192):** a tensor-core decode variant for G = 5 is prototyped before the 14B
  template. It has its own arithmetic, and the 14B carries its own T8 anyway.
- **Otherwise:** CUDA cores; the question returns at the kernel-only bench, where G = 5 is measured directly.

**Reading 4, the int8 design, from Q.**

- **Q ≥ 0.85:** nothing changes.
- **Q < 0.60:** the R1 dequantization cost (attention.md §3.5) is recounted, and the int8 entry point is timed on the
  kernel-only bench before build step 6.
- **Otherwise:** nothing changes.

## 7. Deviations

Any departure from this file goes in `proofs/preregistration-fi-decode-anchor-<date>-ECARTS.md`, beside it.

## 8. Cost

*Estimated* on l40sx1 at $1.80/h. Image pull 1 to 2 min: `awq-vllm-4b` ran 226 s in all with the same image and 10 GB
of checkpoints (`docs/data/jobs.csv`). FlashInfer's JIT, 0 to 8 min, for up to four modules (CUDA-core and tensor-core,
f16 and e4m3) if `flashinfer-jit-cache` does not hold them; the first-call times report which. The bench, 2 to 3 min:
24 (L, arm) pairs, 9 rounds of passes under 2 ms, graph captures, references and one profiler pass each. In all 5 to
14 min, **$0.15 to $0.42**. Timeout 25 min, **$0.75** at worst.

## 9. What this run does not establish

- Our kernel's η. FlashInfer's exp2, merge, head mapping and tiles differ from ours (§1).
- The cost of our polynomial exp2, which the kernel-only bench times.
- G = 5 on CUDA cores, which this version of FlashInfer does not dispatch.
- Our int8 store: e4m3 with one scale per tensor is a proxy for an 8-bit read, not for q8 g64 with two f16
  coefficients a group.
- Quality of any kind, any other card, and any engine-level figure.
