# Preregistration. How fast FlashInfer's batch-1 GQA decode reads a Qwen3-4B f16 KV cache on our L40S

**Written on 2026-11-04, committed and TIMESTAMPED BEFORE the run.** Operator go: given in chat on 2026-11-04 for the first wave of the throughput validation roadmap, under a $10 HF cap for that wave. Cost: about **$0.12 to $0.30** on l40sx1, *estimated* in §8; timeout 20 min, so **$0.60** at worst (*computed*). Cap for this wave: **$10** (operator, 2026-11-04), shared with the other jobs of the wave, whose own preregs carry their costs. Spent before it: $248.84 over 237 rows of the registry (*measured*, `docs/data/jobs.csv`), plus $3.19 for five billed jobs the registry misses (*computed* on their measured durations at $1.80/h: `6aaf9e00`, `6aafac81` and `6aafc951` of 2026-09-20, `6aae7666` of 2026-09-19, `6ab087f2` of 2026-09-21), so $252.03 billed before this run (*computed*).

## 1. Why this run exists

The 4B's decode attention reads its f16 cache at 0.259, 0.252 and 0.256 of 864 GB/s at 2,048, 4,096 and 8,192 positions (*computed* from `docs/mesures/served-grouped-2026-11-02.txt`). The formula is η_marg(L) = 147,456 × L / ((t(L) − t(5)) × 864 GB/s), on the grouped config's step times 7.43, 8.78, 10.20 and 12.90 ms. 147,456 B is the cache one position adds over the 36 layers (36 × 2 × 8 × 128 × 2 B).

A split-KV decode kernel of our own is the attention lead's main build item, about 1.5 to 2.5 weeks for its CUDA part (*estimated*). Its gain at 8k rests on one unmeasured number: the read rate η such a kernel reaches on Ada at batch 1 with G = 4. The lead's planning band is [0.60, 0.85], *estimated*, with no Ada source for a GQA decode kernel. The nearest measured figure is a different access pattern: cuBLAS reads FP16 weights at 672 GB/s in a matvec, 0.78 of 864 (*measured*, `docs/format-noyau.md` §6, `nullk-plancher-2026-08-16.txt`).

FlashInfer's CUDA-core decode is the closest kernel that exists. At f16 and head dim 128 it gives each query head 16 lanes of 8 dims, and it splits the cache into chunks of at least 256 positions merged by a second kernel (`include/flashinfer/attention/decode.cuh`, `SingleDecodeWithKVCacheDispatched`, at tag v0.6.14). The lead's design shares both choices. This run measures FlashInfer's η on our card and our geometry, with a cold L2 and no host work.

The outcome changes three concrete things (§7): the η band every projection of the lead uses, the proposal to the operator on whether the CUDA kernel is written now, and the mapping and core set that kernel's first version carries. It pronounces no kill (METHODE §1).

## 2. What runs

One job on l40sx1, launched by `ops/jobs/fi-decode-anchor.sh` through `ops/run.py bench`.

- **Image.** `vllm/vllm-openai:v0.26.0`, the tag every vLLM job of the repository ran, at manifest digest `sha256:ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52` (recorded in `docs/mesures/awq-vllm-4b-2026-08-17.txt`). On 2026-11-04 Docker Hub resolved the tag to that digest, pushed 2026-07-25 (*measured*, the `DRY_RUN` of the job script). The launcher resolves the tag again and refuses when the digest moved.
- **Packages, pinned by version and gated in the job.** At vLLM tag v0.26.0, `requirements/cuda.txt` pins `torch==2.11.0`, `flashinfer-python==0.6.14` and `flashinfer-cubin==0.6.14`. `docker/Dockerfile` pins `CUDA_VERSION=13.0.2`, installs `flashinfer-jit-cache==0.6.14` from the cu130 index, and installs `cuda-nvcc` for FlashInfer's runtime JIT (both files read on 2026-11-04). Nothing is pip-installed. The job prints and gates these versions before anything is timed (§3, gate 2), and keeps `pip freeze`.
- **Scripts.** The bench, `ops/fi_decode_bench.py`, is uploaded to `hf://buckets/Pier-Jean/jobs-artifacts/fi-decode-anchor-2026-11-04/` and checked by sha256 in the job: `6f08b7a53d3aefb7d0fc658194e28606e44f569aaf9625f778fc676caf09f04a` at this commit. The job script itself goes inline to `run_job` as `bash -lc`. No Space is built, so `UPLOAD_ALLOW` and `ops/Dockerfile.cuda` are not involved. No code of ours runs, so no `oracle` (rule 10 applies to our forward pass).
- **Object.** For each arm and each length L in {2,048, 4,096, 8,192}: 36 distinct per-layer KV sets of the 4B geometry, 8 KV heads, head dim 128, f16, layout HND `[8, L, 128]`, random normal values. One query of 32 heads, G = 4.
- **One pass** is 36 calls of `flashinfer.single_decode_with_kv_cache`, one a layer. It stands for one decode step's attention cache read at batch 1, and it reuses no set.
- **Byte accounting.** A pass reads K and V of all 36 layers: 36 × 2 × 8 × L × 128 × 2 B = 147,456 × L B, so 0.302, 0.604 and 1.208 GB (*computed*). The 32 × 128 query and the output are not counted, below 0.1 % at 2,048. A layer call reads 4,096 × L B, and the pass time is the step's attention time over 36 layer calls, launch gaps inside the graph included.
- **Arms.** `P`, the CUDA-core decode kernel (`use_tensor_cores=False`). `T4`, the tensor-core path at the same G (`use_tensor_cores=True`, the FA2 single-prefill kernel).
- **Timing.** The pass is captured once as a CUDA graph and timed by CUDA events around each replay. Before every timed replay the job reads a 256 MB buffer, above the 96 MB L2, outside the timed window. Two rounds are discarded and seven kept. In every round each (arm, L) runs 10 replays in one fixed order, and the round's value is the median of its 10. A pass that cannot be captured runs eagerly and is flagged; its value is then the profiler's kernel-only sum, one profiled pass a round. One profiled pass per (arm, L) also gives the kernel names and the kernel-only sum.
- **Check.** Set 0 of every (arm, L) is compared with an f32 torch reference of the same attention.

Quantities, *computed* after the run, with medians over the seven rounds and ranges from the slowest and the fastest round. Ratios are formed round by round, since every configuration runs in every round.

- η(arm, L) = 147,456 × L / (t × 864 GB/s).
- **A8** = η(P, 8,192), the anchor.
- **F** = η(P, 2,048) / η(P, 8,192), the fill sensitivity.
- **Rtc** = η(T4, 8,192) / η(P, 8,192), the core comparison.
- **S_P** = 147,456 × 6,144 / ((t(P, 8,192) − t(P, 2,048)) × 864 GB/s), the marginal rate between the two lengths, the same form as our 0.25 to 0.26.

η is a fraction of the same card's peak, 864 GB/s on the L40S. No speed of ours is divided by FlashInfer's or by vLLM's (house rule 5). The anchor is compared with our attention in GB/s and in η on one card only.

## 3. Gates, each one voids the run

1. `nvidia-smi` names an L40S.
2. The job prints `vllm 0.26.0`, `torch 2.11.0` (with or without `+cu130`), `torch.cuda 13.0`, `flashinfer-python 0.6.14`, `flashinfer-cubin 0.6.14` and `flashinfer-jit-cache 0.6.14` (with or without `+cu130`), and the bench's sha256 matches §2.
3. Arm P is timed at all three lengths, and its set-0 error is at most 2e-2 of max |ref|.
4. `fi-decode.json` holds every configuration, timed or marked void with its exception.

If arm P was timed eagerly, its readings use the profiler's kernel-only sums, declared here before the run.

## 4. Outcomes reported, never gated

Every configuration's medians, ranges and η; S_P; the kernel names and kernel-only sums, and η on the kernel-only sum; the first-call times, which tell a JIT build from a cache load; whether each pass was captured; T4's voids and errors, if any.

## 5. Signed predictions

| quantity | point | interval |
|---|---|---|
| A8 = η(P, 8,192) | **0.65** | [0.45, 0.85] |
| η(P, 4,096) | **0.52** | [0.30, 0.78] |
| η(P, 2,048) | **0.36** | [0.18, 0.62] |
| F | **0.55** | [0.30, 0.85] |
| S_P | **0.75** | [0.50, 0.95] |
| Rtc | **0.85** | [0.25, 1.20] |
| P's pass captured as a graph at every length | **yes** | |
| P's profile at 4,096 and 8,192 shows a decode kernel and a merge kernel per call | **yes** | |

Reasoning, *estimated*. At 8,192 positions the chunk rule gives 32 chunks of 256 a KV head, so 256 CTAs of 128 threads on 142 SMs, each with two shared-memory stages of 8 positions in flight (stage count read at FlashInfer `main`, not at the tag). That keeps about 2 MB in flight against roughly 0.6 MB that 864 GB/s needs at 0.7 µs of latency, so the read can approach the matvec's 0.78. Two kernels a call cost a few microseconds a layer, against an ideal pass of 1.40 ms at 8,192 and 0.35 ms at 2,048 (*computed* at 864 GB/s). At 2,048 only 64 CTAs run, under one per SM, and the fixed cost weighs four times more, hence F well under 1. The tensor-core path runs 4 valid rows of 16 at batch 1, and whether its single-prefill kernel splits the cache as finely is not known, hence Rtc's wide interval. FlashInfer reports near-full bandwidth for MHA decode on the RTX 6000 Ada, the same AD102 chip, and "40%+" for its CUDA-core GQA decode on A100 and H100, whose FP32 is weaker (FlashInfer blog, read during the wave-5 review).

## 6. What the outcomes feed

Every projection below uses one formula, *computed*: step(η) = 7.43 ms + 147,456 × 8,192 B / (η × 864 GB/s) at 8,192 positions. It keeps today's 5-token step (*measured*, `served-grouped-2026-11-02.txt`) and replaces the cache-dependent part by a read at η. It credits no launch removed at short context, so it is conservative. At η = 0.256 it gives 77.6 tok/s against the 77.5 measured.

| η | 0.25 | 0.30 | 0.40 | 0.50 | 0.60 | 0.75 | 0.85 |
|---|---|---|---|---|---|---|---|
| 8k decode, tok/s | 76.8 | 82.7 | 91.5 | 97.8 | 102.5 | 107.6 | 110.2 |
| × on today's 77.5 | 0.99 | 1.07 | 1.18 | 1.26 | 1.32 | 1.39 | 1.42 |

Our kernel is assumed to land near FlashInfer's η, because it shares the lane partition and the split floor (§1). It could do better by folding the merge into the last CTA, or worse through its 13-instruction exp2 polynomial. The kernel-only bench measures our own η_k after the kernel exists.

## 7. Decision rules

Three readings, each evaluated alone. Each one names the concrete decision it changes. None pronounces a kill: a kill is the operator's decision on a fundamental criterion (METHODE §1). A reading reads the median. When its round range crosses a threshold, the proposal names both lines, the median's line first.

**Reading 1, the η band and whether the CUDA kernel is written now, from A8.**

- **A8 ≥ 0.60.** The band stays [0.60, 0.85]. The 8k projection stays 102.5 to 110.2 tok/s, ×1.32 to ×1.42 (§6). Proposal to the operator: write the CPU replay and the CUDA kernel now. The kernel-only bench prereg takes A8 as its point prediction for η_k.
- **0.40 ≤ A8 < 0.60.** The band becomes [0.40, 0.60], and the 8k projection 91.5 to 102.5 tok/s, ×1.18 to ×1.32. Every 8k figure of the attention lead is recomputed on that band before any further prereg of the lead is stamped. Proposal: write the CPU replay now, at $0. The operator weighs the CUDA kernel, 1.5 to 2.5 weeks, against ×1.18 to ×1.32 at 8k before it is started.
- **A8 < 0.40.** The band becomes [0.25, 0.40], and the 8k projection 76.8 to 91.5 tok/s, ×0.99 to ×1.18. Proposal: do not write the CUDA kernel now. The CPU replay may still be written, at $0, if another lead needs it. The proposed reopening condition: a decode kernel measured on Ada above 0.40 at G = 4 and batch 1, from any source.
- **Otherwise** (a gate failed, or A8 not formed): not settled. The band stays as it is, the raw output is kept, and the operator decides on a rerun under this prereg, with its deviation written beside it.

**Reading 2, the mapping set of the kernel's first version, from F.** It applies only if the CUDA kernel is written.

- **F ≥ 0.80.** Fill does not limit FlashInfer at 2,048. The first version carries the CTA-per-split mapping only.
- **F < 0.80.** Fill limits it. The first version also carries a CTA-per-unit mapping, four times the CTAs at 2,048 and bit-neutral, and the kernel-only bench times both.
- **Otherwise** (F not formed): both mappings are written.

**Reading 3, CUDA cores or tensor cores at G = 4, from Rtc.** It applies only if the CUDA kernel is written.

- **Rtc ≥ 1.15.** The tensor-core path reads the cache faster at batch 1. A tensor-core decode variant at G = 4 is added to the first version, and the kernel-only bench times it against the CUDA-core one.
- **Rtc < 1.15.** CUDA cores only, as designed.
- **Otherwise** (T4 void at 8,192, or Rtc not formed): CUDA cores only, and the void is reported.

## 8. Cost

*Estimated* on l40sx1 at $1.80/h. The vLLM jobs on this image billed 4 to 5 min each, image pull and checkpoint download included: `awq-speed-4b` 4 min with 226 s running, `paper-awq-speed-8b` 4 min, `gsm8k-vllm-4b-r2` 5 min (*measured*, `docs/data/jobs.csv`). This job downloads no checkpoint.

- Pull and start, 1 to 3 min.
- FlashInfer modules: 0 min if `flashinfer-jit-cache` holds them, up to about 5 min to build the decode and the single-prefill module with nvcc on 8 vCPUs. The first-call times report which.
- Bench, under 2 min: 6 configurations × 9 rounds × 10 replays of at most about 4 ms, the references, 6 graph captures and 6 profiled passes, and 1.2 GB of pools.

In all 4 to 10 min, **$0.12 to $0.30** (*computed* on *estimated* durations). Timeout 20 min, **$0.60** at worst (*computed*). A retry is a second launch at the same ceiling, under the operator's go.

## 9. What this run does not establish

- Our kernel's η. FlashInfer's exp2, merge kernel, head mapping and stage sizes differ from the lead's design.
- The effect of any kernel on the served decode step. The in-situ A/B measures that, after the kernel exists.
- G = 5, the 14B's ratio, which FlashInfer 0.6.14's CUDA-core decode does not dispatch (`DISPATCH_GQA_GROUP_SIZE` takes 1, 2, 3, 4 and 8).
- Any 8-bit cache, any card but the L40S, any engine-level figure, and any quality.
- Real cache contents: random normal values change no byte count and no instruction of a decode kernel.

## 10. Deviations

Any departure from this file goes in `proofs/preregistration-fi-decode-anchor-2026-11-04-ECARTS.md`, beside it, never inside it.
