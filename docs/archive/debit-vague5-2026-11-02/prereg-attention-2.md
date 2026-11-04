DRAFT, not stamped, not committed

# Preregistration. The split-KV decode kernel against the served grouped attention, in the model

**Draft written on 2026-11-02, to be committed and TIMESTAMPED BEFORE the run.** Operator go to launch: **not given**.
It needs build steps 1 to 3 of attention.md §6 and the freeze of `prereg-attention-2a` §6, rule 3. Cost: about
**$0.39 to $0.48** on l40sx1, *estimated* below (§8), timeout 45 min, so $1.35 at worst. Cap asked: **$2**, which covers
one retry. Spent before it: $248.84 over 237 rows of the registry, plus what `prereg-attention-1` and `-2a` spend.

## 1. Why this run exists

The 4B decodes at 134.5, 113.8, 98.0 and 77.5 tok/s after 5, 2,048, 4,096 and 8,192 tokens under the grouped form
(*measured*, `docs/mesures/served-grouped-2026-11-02.txt`). Above the 5-token step, its context-dependent work runs at
about 220 GB/s, η 0.25 to 0.26 at all three lengths (*computed*, same medians). At 8,192 that is 5.47 ms of a 12.90 ms
step. The grouped form runs six launches a layer for attention: two `slice_set`, two cuBLAS matmuls, the f16 scale and
candle's softmax.

The commit that carries this file serves `attn_dec`, one split-KV kernel per layer, with the canonical arithmetic
frozen after the kernel-only bench (`prereg-attention-2a`) and the mapping that bench chose. This run asks what the
kernel does to the decode step of the model. Throughput is a fundamental criterion (METHODE §1), so this run carries
the speed gate of the lead. It is not a quality control: the kernel is not bit for bit with cuBLAS, and its licence is
a T8 control on its own prereg (V6).

## 2. What the code holds, and what the Mac showed

To be filled at commit time with *measured* host facts: the CPU replay and its tests, the host probe of the kernel
text, the mutants killed, the frozen constants and their commit, the bench figures of `prereg-attention-2a` with their
journal, and the load-time card checks the kernel runs before any token.

## 3. What runs

One image built from the commit that carries this file. One job, l40sx1, the sealed 4B
(`sealed-4b-2026-09-23/qwen3-4b-sealed.bin`, sha256 `886391a8…`), config `configs/qwen3-4b-tetra-e4-grouped.json`
(sha256 `2455fe85…`), flags spelled out on each step. `ops/jobs/attn-ab.sh`.

**The prefill path.** Both arms prefill through the path served at stamp time, named here before the stamp: today
`prefill_chunked` with the rows kernel, 4,096 rows a call, and cuBLAS grouped attention; the GEMM route instead if the
operator's branch has served it by then. Rows of more than 8 never reach the kernel, so the two arms' caches are
equal after the prefill. The path is the same in both arms, or the run is void (gate 7).

1. **Oracle** on Qwen3-0.6B, rule 10.
2. **The bench door replayed**, `fusedrun <file> 256` with the served flags, against the dense arm: gate 4.
3. **Diagnostic re-bench**, `LLVQ_ATTN_BENCH=8192` on the chosen variant only, k = 1 and 8, so that η_k comes from this
   image. Reported, never gated.
4. **The in-situ A/B**, `LLVQ_ATTN_AB=5,2048,4096,8192`, 256 new tokens, cudarc's events off. Arms `grow(256)+grouped`
   (served) and `grow(256)+kernel`. For each length: one prefill per arm in chunks of 4,096, outside every clock; the
   first decode step's logits compared; then 1 discarded and 5 timed decodes of 256 tokens per arm, interleaved, each
   cache truncated back to the prompt before each decode. Speed `255 / t`, ratios round by round, texts compared, the
   first differing token printed.

## 4. Gates, each one voids the run

1. The image builds, and `oracle` prints MATCH.
2. The sealed file and the config at their bytes and sha256 on the mount.
3. Every load prints `int4: tv_q4_hh equals tv_q4_h bit for bit`.
4. Step 2 prints `256 tokens identical to the dense arm`, and its decoded line hashes to `8784ffc4…`.
5. Every load prints the kernel's load-time check: rows invariant and equal to the CPU replay at
   L ∈ {1, 255, 256, 257, 1000, 4096} and k ∈ {1, 3, 8}, with a grid of `s_max` CTAs as well as the host's count.
6. Step 4 prints `cudarc event tracking: off (one stream, ordered by issue)` and both tables in full.
7. Step 4 prints the prefill path of each arm, and the two are the same.
8. The header of every step reads `tile 64`, and no `LLVQ_*` variable is in the job environment.

## 5. Signed predictions

| quantity | point | interval |
|---|---|---|
| in situ, kernel / grouped at 5 / 2,048 / 4,096 / 8,192 | **1.04 / 1.15 / 1.25 / 1.41** | [0.99, 1.07] / [1.06, 1.24] / [1.12, 1.36] / [1.20, 1.56] |
| in situ, kernel decode tok/s at 8,192 | **110** | [94, 121] |
| re-bench η_k(8,192), chosen variant | **0.70** | [0.45, 0.90] |
| first decode step, max \|dlogit\| against grouped | **0.05** | [0.005, 0.5] |
| texts identical to the grouped arm's | **at 5 tokens, and at 3 of the 4 lengths at least** | |

Reasoning, *estimated* (`calc/attention/attention_v2.py`, sections 2 and 3). Removing five of the six attention
launches a layer saves about 0.3 ms a token at 2.1 to 2.3 µs a launch (*computed* from the trims and gqa-ab journals);
if the branch's fused prologue already writes K and V, the kernel removes three and the 5-token ratio falls to about
1.02. The step is then t(L) = 7.12 ms + 147,456 × L B / (η × 864 GB/s). The known flaws: η is the bench's; the
combine and the tickets add a fixed cost the model ignores.

## 6. Decision rules

On the in-situ ratios only, formed round by round. Evaluated in this order; a result takes the first rule that matches.

1. **An identity check fails** (rows of one call against separate calls, card against CPU replay, or the two grids):
   the kernel has a bug. No speed figure of this run feeds a decision. Fix, then a deviation file and a rerun on the
   operator's go.
2. **Stop proposal:** the in-situ ratio range at 8,192 lies entirely below 1.00, or the in-situ median at 5 tokens is
   below 0.97. The proposal goes to the operator after at most one revision of the bit-neutral mapping; the kill is
   his.
3. **Pass:** the low end of the in-situ range at 8,192 is at least 1.15, and the in-situ median at 5 tokens is at least
   0.99. The T8 control prereg (attention.md §5, V6) is written; it needs its own go.
4. **Middle:** every other result with all gates passed. One revision of the mapping before any T8 control, if the
   operator spends it. Its target is diagnosed from the re-bench: η_k(8,192) below 0.60 points to the memory pipeline
   (stages, depth, CTAs per SM); at or above 0.60 it points to the fixed costs (launches, combine, prologue).
5. **Otherwise** (a gate failed, a table is missing): not settled; the raw output is kept and the operator decides.

The 2,048 and 4,096 ratios and the re-bench are reported beside the rule, never in it. Text identity is reported,
never gated: the kernel computes the softmax in f32 where the grouped form computes it in f16, so a divergence is
expected somewhere and belongs to the T8 control.

## 7. Deviations

Any departure from this file goes in `proofs/preregistration-attn-ab-<date>-ECARTS.md`, beside it.

## 8. Cost

*Estimated* on l40sx1 at $1.80/h. Container and image pull 3 min, oracle 2 min, step 2 about 3 min, step 3 under
1 min. Step 4: two prefills per length, 0, 16, 34 and 70 s on today's prefill path, then 12 decodes of 256 tokens per
length at 7 to 13 ms a token, about 5 min. In all 13 to 16 min, **$0.39 to $0.48**. Anchors: `gqa-ab` ran three arms at
four lengths in 575 s for $0.29; `served-grouped` ran 1,082 s for $0.54 (`docs/data/jobs.csv`). Timeout 45 min,
**$1.35** at worst.

## 9. What this run does not establish

- Quality. A paired perplexity by teacher-forced decode on fresh windows at 4,096 and 8,192, with a third arm for the
  q8 cache under the kernel, and GSM8K through the kernel, are the T8 control (V6, V7). Not run here.
- The prefill: the kernel serves rows of at most 8.
- The int8 store, the 8B and 14B models in situ, and any other card.
- Where the remaining time of the step goes: no kernel timeline in the image.
