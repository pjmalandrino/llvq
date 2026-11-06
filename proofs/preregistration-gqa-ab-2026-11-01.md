# Preregistration. Attention over the cached heads without `repeat_kv`, from 5 to 8,192 tokens of prompt

**Written on 2026-11-01, to be committed and TIMESTAMPED BEFORE the run.** Operator go to launch given 2026-11-01 ("Go, plafond 2 $"). Cost: about
**$0.54** on l40sx1, *estimated* below (§8), timeout 45 min, so $1.35 at worst. Cap for this wave: **$2**,
which covers one retry. Spent before it: $247.10 over 228 priced jobs (*measured*, `docs/data/jobs.csv`).

## 1. Why this run exists

The run of 2026-10-30 (`docs/mesures/trims-contexts-2026-10-30.txt`) measured the served decode at 122.5, 82.7, 61.4
and 33.2 tok/s after prompts of 5, 2,048, 4,096 and 8,192 tokens. At 8,192, 72.9 % of a decode token goes above the
5-token step (*computed* from those medians). By that prereg's rule, the KV path is the first lever.

On the served path (`KvStore::Cat`, expanded heads), every decode step does, in each of the 36 layers (*counted* in
the code, *computed* in bytes for Qwen3-4B in f16, 8 key heads of 128):

| work | launches | bytes a cached position |
|---|---|---|
| `Tensor::cat` of the history, K and V | 4 | 8 KB read and written |
| `repeat_kv` of K and V, a `cat` of 4 copies each | 8 | 32 KB read and written |
| the two attention matmuls, on the expanded heads | 2 | 16 KB read |
| scale and softmax | 2 | small |

That is 56 KiB a position and a layer, 2.06 MB a position over the model: 16.9 GB a token at 8,192, against 1.38 GB
of weights.

## 2. What the code now holds, and what the Mac showed

The commit that carries this file adds two settings, both off by default, so every published path runs as before:

- `kvq::AttnHeads`, how attention reads the `n_kv` cached heads. `Expanded` is the published form. `Shared` runs one
  matmul per query head of a group against the stored key head, each of the expanded form's shape, with no copy.
  `Grouped` stacks the 4 query heads of a key head as the rows of one matrix: one matmul per key head.
- `KvStore::Grow(chunk)`, a buffer grown by whole chunks of 256 positions and written in place. Attention reads a
  view of the valid prefix, whose batch stride candle passes to cuBLAS. The history is copied once a chunk, not once
  a step.

Per layer and decode step (*counted*): `Grow` plus `Grouped` launches 6 kernels and reads 4 KB a cached position;
`Grow` plus `Shared` launches 20 and reads 16 KB a position, of which 12 KB are re-reads of a key or value head that
the previous matmul just read (a head at 8,192 positions is 2 MB; the L40S has 96 MB of L2).

*Measured* on the Mac CPU, in the tests of `llvq-llm/src/model.rs`, on tiny random models with groups of 2 and of 4:

- `Grow` alone, and `Shared` on either store, return the published form's logits bit for bit, in f32 and f16, in a
  chunked prefill, in 20 decode steps and in the uncached full-window forward (`attention_forms_agree`).
- `Grouped` moves the decode logits by 4.5e-8 to 6.0e-8 in f32 and 2.4e-4 to 3.7e-4 in f16. Its prefill came out
  bit for bit. On a random model whose logits tie often, one such difference flipped a greedy token at the ninth
  step.
- 13 mutants of the new code, each killed by a test.

The CPU says nothing about cuBLAS. Whether `Shared` stays bit for bit on the card depends on cuBLAS rounding a batch
element the same way at batch counts of 8 and 32.

## 3. What runs

One image built from the commit that carries this file. One job, l40sx1, the sealed 4B
(`sealed-4b-2026-09-23/qwen3-4b-sealed.bin`, sha256 `886391a8…`), config `configs/qwen3-4b-tetra-e4.json`
(`ops/jobs/gqa-ab.sh`).

1. **Oracle** on Qwen3-0.6B, rule 10.
2. **The published arm, replayed.** `fusedrun <file> 256` with the served flags spelled out and `LLVQ_EMBED=q4`,
   against the dense arm of the same process, as in run 1 of the paper table and step 2 of the run of 2026-10-30.
   It checks that the default path did not move: `forward_cached` was restructured around the new forms.
3. **The A/B.** The same flags plus `LLVQ_GQA_AB=5,2048,4096,8192`, 256 new tokens, cudarc's events off as on the
   served door. Three arms in one model: `cat+expanded` (published), `grow(256)+shared`, `grow(256)+grouped`. For
   each prompt length: one prefill per arm in chunks of 4,096, outside every clock; the logits of the prompt's last
   position and of the first decode step, compared bit for bit with the published arm's; then 1 discarded and 5
   timed decodes of 256 tokens per arm, every arm in every round, each cache truncated back to the prompt before
   each decode. Speed is `255 / t`, the reading of `LLVQ_CONTEXTS`. Ratios are formed round by round.

The timed rounds of the `Grow` arms reuse buffers that their first decode grew. In service, one chunk reallocation
copies the cache once every 256 tokens: at 8,192 positions, about 0.1 % of a decode token (*computed*).

## 4. Gates, each one voids the run

1. The image builds, and `oracle` prints MATCH.
2. The sealed file and the config at their bytes and sha256 on the mount.
3. Every load prints `int4: tv_q4_hh equals tv_q4_h bit for bit`.
4. Step 2 prints `256 tokens identical to the dense arm`, and its decoded line hashes to `8784ffc4…`, the line of
   paper-table run 1.
5. Step 3 prints `cudarc event tracking: off (one stream, ordered by issue)` and its four tables in full.
   `run_gqa_ab` refuses to print if one arm returns a different text in two rounds.
6. The header of every step reads `tile 64`, and no `LLVQ_*` variable is in the job environment.

## 5. Outcomes reported, never gated

For each unexpanded arm and each length: whether its logits are the published arm's bit for bit (prompt, then first
decode step), else the largest difference; and whether its 256 tokens are the published arm's, else the first that
differs.

## 6. Signed predictions

| quantity | point | interval |
|---|---|---|
| step 2, fused tok/s | **120** | [113, 126] |
| `cat+expanded` decode tok/s at 5 / 2,048 / 4,096 / 8,192 | **122 / 83 / 61 / 33** | [116, 128] / [79, 87] / [58, 65] / [31.5, 35] |
| `grouped / expanded` at 5 | **1.10** | [1.03, 1.18] |
| `grouped / expanded` at 2,048 | **1.45** | [1.20, 1.65] |
| `grouped / expanded` at 4,096 | **1.85** | [1.45, 2.15] |
| `grouped / expanded` at 8,192 | **2.9** | [1.9, 3.6] |
| `shared / expanded` at 5 | **0.94** | [0.85, 1.00] |
| `shared / expanded` at 2,048 | **1.20** | [1.02, 1.40] |
| `shared / expanded` at 4,096 | **1.45** | [1.15, 1.75] |
| `shared / expanded` at 8,192 | **2.2** | [1.5, 3.0] |
| `shared` logits, both checks, at all four lengths | **bit for bit** | |
| `grouped`, first decode step | **not bit for bit**, max \|dlogit\| | [0.002, 0.25] |
| `grouped` text at 5 tokens | **identical** | |

Reasoning, *computed*. Launches: about 2.3 µs each, from the cast trim of 2026-10-30 (84 launches, 0.19 ms).
`Grouped` drops 10 copies a layer, 360 a token, about 0.8 ms of 8.2 at 5 tokens. `Shared` adds 6 matmuls and drops 2
copies a layer; a cuBLAS call costs more host time than a copy, hence a loss at 5 tokens. Above 5 tokens, the
published step pays 3.9, 8.1 and 22.0 ms at 2,048, 4,096 and 8,192 (*measured*, 2026-10-30) for its 2.0 MB a
position. `Grouped` reads 7 % of those bytes, but its value matmul (128 by 4 by L, 8 batches) fills few of the 142
SMs unless cuBLAS splits it, hence the wide intervals. `Shared` reads 29 % of them, most from L2, through four
times as many matmuls.

## 7. Decision rules

- **A, `shared` changes no output.** If `grow(256)+shared` is bit for bit at all four lengths, in both logit checks,
  and its text is identical at all four: it becomes the served attention in a follow-up commit, with no change to
  any gate, provided its ratio range lies above 1.000 at 2,048, 4,096 and 8,192 and its median at 5 tokens is at or
  above 0.97. If it loses more than 3 % at 5 tokens, the operator chooses between the published measurement point
  and the product's context.
- **B, `grouped`.** If it too is bit for bit everywhere, rule A applies to it, and the faster form is served.
  Otherwise it is not served on this run's strength. Its ratios and differences are the input to decision T8, the
  operator's: whether a bounded tolerance replaces bit identity as the gate.
- **C.** An unexpanded arm whose ratio range at 8,192 lies below 1.000 is not served, and a kernel timeline (`nsys`)
  comes before any further step on this path.
- If `shared` is not bit for bit on the card, both forms fall under T8.

## 8. Cost

*Estimated* on l40sx1 at $1.80/h. Container and image pull 3 min, oracle 2 min, step 2 about 3 min. Step 3: three
prefills per length, 0, 25, 51 and 105 s; then 18 decodes of 256 tokens per length, about 40, 50, 60 and 90 s. With
loading, about 8 min. In all about 18 min, **$0.54**. Timeout 45 min, **$1.35** at worst.

## 9. What this run does not establish

- Quality beyond these texts: if `grouped` is not bit for bit, a paired perplexity or MMLU through the kernel would
  be the next measurement, under T8.
- Anything at 8B or 14B, on another card, or on Metal (the Mac CPU result above is not a Metal result).
- The step after this one: an int8 cache, read by a decode attention kernel.
- Where the time of the remaining launches goes: no kernel timeline in the image.
