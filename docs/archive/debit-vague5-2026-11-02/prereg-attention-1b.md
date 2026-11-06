DRAFT, not stamped, not committed

# Preregistration. A mature engine's decode slope and prefill time at depth on our L40S (optional)

**Draft written on 2026-11-02, to be committed and TIMESTAMPED BEFORE the run.** Operator go to launch: **not given**.
**Optional:** no action of the attention lead depends on this run; the FlashInfer anchor (`prereg-attention-1`) sets
the η band. Cost: about **$0.12** on l40sx1, *estimated* below (§8), timeout 20 min, so $0.60 at worst. Cap asked:
**$1**, which covers one retry. Spent before it: $248.84 over 237 rows of the registry (*computed* from
`docs/data/jobs.csv` on 2026-11-02), plus what `prereg-attention-1` spends.

## 1. Why this run exists

The FlashInfer anchor measures one kernel with a cold L2 and no host work. This run measures a whole engine on the same
card and the same model: llama.cpp's decode time at four depths, whose slope is an engine-level KV read rate, and its
prefill rate at 2,048 and 8,192 tokens, a time to the first token. Two decisions can use those figures, and neither
belongs to this lead: the operator's choice between the prefill options P1 and P2 (attention.md §3.8, decision 9 of the
program report), and the program's engine comparison, stated in GB/s (rule 5).

llama.cpp master `cd8427d` (2026-11-02) sends batch-1 decode with an f16 or q8_0 cache to its CUDA-core vector kernel
on Ada when the GQA ratio is 4 (`ggml/src/ggml-cuda/fattn.cu`, lines 656 to 665). That kernel runs one CTA per query
head (`fattn-vec.cuh`, lines 104 to 111) on position-major cache rows. The slope also carries the engine's per-token
host and mask work, which this run cannot separate.

## 2. What runs

One job, l40sx1, image `ghcr.io/ggml-org/llama.cpp` **pinned by the digest of `full-cuda` read at stamp time** (the
command is in the job script) and written here before the stamp. If HF Jobs refuses a digest reference, the tag is
passed instead and §3 gate 3 carries the reading. The two GGUFs of `m4-iq2-cuda-2026-08-30` already in the bucket
(`m4-iq2-cuda-1b57b7d3/`), checked by sha256: `qwen3-4b-iq2xxs.gguf` (`19a8ed49…`) and `qwen3-4b-f16.gguf`
(`259393f9…`). Script `ops/jobs/attn-depth-anchor.sh` (draft beside this file, in `code/attention/`).

Known infra failure: this image failed twice at container start on 2026-09-25 (OCI runtime create failed,
`MOUNT_ATTR_IDMAP` on `/usr/bin/nvidia-cuda-mps-control`, $0 each, registry rows `paper-iq2-census (echec infra)`) and
passed an hour later. Such a failure bills nothing; the retry waits an hour.

- **A.** `llama-bench -m <iq2xxs> -m <f16> -ngl 99 -fa 1 -p 0 -n 128 -d 0,2048,4096,8192 -r 5 -o jsonl`. Decode of 128
  tokens after a context of d tokens, f16 cache.
- **B.** `llama-bench -m <iq2xxs> -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 0 -n 128 -d 0,2048,4096,8192 -r 5 -o jsonl`. The
  same with a q8_0 cache.
- **C.** `llama-bench -m <iq2xxs> -m <f16> -ngl 99 -fa 1 -p 2048,8192 -n 0 -r 3 -o jsonl`. Prefill.

Derived quantities, *computed* after the run, for each GGUF and cache type:

- t(d) = 1 / tg128(d), per repetition, from `samples_ns`;
- the endpoint slope s = (t(8192) − t(0)) / 8192, and the least-squares slope over the four depths;
- the engine-level KV read rate R = bytes per position / s, with 147,456 B for the f16 cache (36 layers × 8 KV heads ×
  128 × 2 × 2 B) and 78,336 B for q8_0 (34 B per 32 values, 8.5 bits a value, the count of our q8 g64);
- the time to the first token at 8,192, 8,192 / pp8192.

Each slope is formed from the median t at each depth, its envelope from the slowest and fastest repetitions. Rates of
two engines are compared in GB/s only (rule 5).

## 3. Gates, each one voids the run

1. Both GGUFs at their sha256 on the mount, and `nvidia-smi` names an L40S.
2. Every JSONL row of A and B reports flash attention on, the requested cache types, and `n_depth` in
   {0, 2048, 4096, 8192}; A has 8 decode rows, B has 4, C has 4.
3. `llama-bench --version` prints a build number and a commit. If the commit is not `cd8427d`, `fattn.cu` is read at
   that commit ($0) before §6 is applied; if its Ada vector-kernel rule differs, the run stays valid as an engine
   figure and the "same kernel as at cd8427d" sentence of the journal is dropped.
4. t(d) increases with d for every GGUF and cache type. A non-monotone series is not a slope.

## 4. Outcomes reported, never gated

tg128 at each depth with its range; both slopes and R for each GGUF and cache type; pp2048 and pp8192 for both GGUFs;
the time to the first token at 8,192. The tg128 at d = 0 is read beside the 312.85 and 84.83 tok/s of
`m4-iq2-cuda-2026-08-30` (another build, without `-fa`): side by side, not divided. R is read beside the FlashInfer
anchor's kernel-level rate, in GB/s.

## 5. Signed predictions

| quantity | point | interval |
|---|---|---|
| IQ2_XXS tg128 at d = 0, tok/s | **310** | [285, 340] |
| f16 tg128 at d = 0, tok/s | **85** | [80, 90] |
| IQ2_XXS tg128 at d = 8,192, f16 cache, tok/s | **190** | [150, 230] |
| f16 tg128 at d = 8,192, f16 cache, tok/s | **72** | [64, 78] |
| R, f16 cache, median of the two GGUFs, endpoint slope, GB/s | **600** | [350, 800] |
| R, q8_0 cache, IQ2_XXS, GB/s (unique bytes) | **450** | [250, 700] |
| IQ2_XXS pp8192, tok/s | **6,000** | [2,500, 12,000] |
| f16 pp8192, tok/s | **10,000** | [5,000, 18,000] |

Reasoning, *estimated*. On this card cuBLAS streams FP16 weights at 672 GB/s (*measured*, `docs/format-noyau.md` §7).
llama.cpp's vector kernel reads each KV head once per query head, so a GQA ratio of 4 multiplies its L2 traffic by 4;
L2 holds a layer's 33.5 MB at 8k, so DRAM traffic should stay near 1×. Its q8_0 path dequantizes V per element, hence a
lower rate in unique bytes. The slope also carries host and mask work.

## 6. Decision rules

No action of the attention lead depends on this run. Its figures go to two readers, each with its own rule.

- **The prefill decision (decision 9).** The time to the first token at 8,192 of both GGUFs is added to the table the
  operator reads when choosing between P1 and P2 (attention.md §3.8), whatever its value.
- **The engine comparison.** R and pp8192 enter the program's engine table in GB/s and tok/s of prefill, beside our
  own measured figures, never divided by them (rule 5).
- **Otherwise** (a gate failed): not settled; raw output kept; the operator decides on a rerun.

## 7. Deviations

Any departure from this file goes in `proofs/preregistration-attn-depth-anchor-<date>-ECARTS.md`, beside it.

## 8. Cost

*Estimated* on l40sx1 at $1.80/h. Container start 1 to 2 min. Five model loads from the mount, about 1 to 2 min, the
8 GB f16 file dominating. A: 48 runs, each a prefill of d tokens and 128 decoded tokens, about 70 s in all. B: about
25 s. C: about 15 s. In all 4 to 5 min billed, **about $0.12-0.15**. Anchor: `m4-iq2-cuda-debit` ran two tg128 tests on
the same GGUFs in 81 s, billed 1 min, $0.04 (`docs/data/jobs.csv`, `hf jobs inspect 6a94a1d445686a1580c19524`).
Timeout 20 min, **$0.60** at worst.

## 9. What this run does not establish

- Our kernel's η, or FlashInfer's: this is another engine, with another thread mapping and its own host work.
- Anything at 8B or 14B. The 14B's GQA ratio of 5 sends llama.cpp to its tensor-core kernel at 8,192 and beyond
  (`fattn.cu` line 664), so a 14B anchor would measure another kernel.
- Quality of a q8_0 cache.
- Any ratio between engines: the comparable quantity is GB/s (rule 5).
