# Preregistration. The attention kernel's T8 control on the sealed 4B: paired decode-path perplexity at 4,096 and 8,192, then GSM8K, against the served grouped form

**Written on 2026-11-04. NOT STAMPED, NOT LAUNCHED.** It is stamped with OpenTimestamps before the first measurement,
and renamed to the day of the stamp if that day is later. Operator go: the one go of lot 1 (`ops/jobs/lot1.sh`), to be
given in chat. Cost: **$0** beyond a few seconds if the condition of §3 fails, **$1.38** if the perplexity band fails,
**$3.27** in full, all on l40sx1 (*estimated*, §9). Ceiling 160 min inside the lot job, **$4.80** (*computed*).
Registry before the lot: $261.35 over 251 rows (*measured*, `docs/data/jobs.csv`, 2026-11-04).

**Base commit:** `02ec809323eb` (branch `integration/lot1`); this file is committed on top of it. **Space sha:**
`142216fc72326a98b94316a7f6cd2e989f0efc50`, built from `02ec809`. The job script is `ops/jobs/attn-t8.sh`; it refuses
to launch without this file and its `.ots`. It runs as the last part of the lot job `lot1-4b`, after the attention
bench r2 (`preregistration-attn-bench-r2-2026-11-04.md`), and only if that part exits 0.

## 1. Why this run exists

- The split-KV decode-attention kernel reads the 4B's cache faster than the served grouped form at long context and
  slower at short context: 95.8 against 78.0 tok/s at 8,192 (×1.23), 126.8 against 136.2 after 5 tokens (×0.93)
  (*measured*, `docs/mesures/attn-bench-2026-11-04.txt`). The fixed-cost revision `c8a01d3` is measured by r2.
- The kernel is not bit for bit with the grouped form. It replaces cuBLAS's scores and candle's f16 softmax by an f32
  online softmax (`docs/archive/debit-vague5-2026-11-02/attention.md` §3.1, "That is the T8 event"). Serving it needs a
  T8 decision of its own: the grouped form's licence does not extend to it (same lead, §7).
- The lead orders the control in two steps (§5, V6 then V7): paired decode-path perplexity at 4,096 on 12 fresh windows
  (61-72) and at 8,192 on 6 windows (7-12), then GSM8K through the kernel paired with a census of the served form.
  This run is V6 with two arms and V7. The third arm of V6, kernel with an int8 cache, waits on an int8 store that is
  not built.

## 2. What the code holds

- `LLVQ_GQA_PPL=<ctx>,<windows>,<first>,kernel` in `bin/fusedrun` (commit `5255470`). It installs the kernel, checked
  against the CPU replay on the card first, and pairs `grow(256)+grouped` with `grow(256)+kernel` window by window
  through `Qwen3::decode_nll_paired`, from window index `<first>`. The decode path feeds one true token at a time and
  needs no prefill call, so 8,192 works. No call of more than 8 rows reaches the kernel, so the prefill path is the
  grouped arm's: the mode checks it bit for bit on the first window when that window fits one prefill call (4,096),
  stops on a difference, and never scores it. The interval is the paired 95 % t interval of `paired_ppl_pct`, the one
  every grouped control used.
- `bin/gsm8k` under a config with `"attention": "kernel"`: `Served::apply` builds the kernel, checks it against the CPU
  replay on the card, and installs it (`served.rs:303-323`). Decode rows go to the kernel; prompts take the grouped
  path.
- The kernel config is the one both attention benches ran, sha256 `d74ea29a…`, written by the job from its launch
  command. It is not a file of `configs/` and changes no served default.
- The kernel text is `llvq-cuda/kernels/attn_dec.cu`, sha256 `c7b0d8d3…`, the text r2 measures, unchanged by the merge
  (`preregistration-attn-bench-r2-2026-11-04-ECARTS.md` E1).
- `oracle`, `fusedrun` and `gsm8k` are in the build list and in the runtime `COPY` of `ops/Dockerfile.cuda`.
- *Computed* on the Mac, $0: the host test `the_kernel_arm_runs_in_the_paired_decode` (`model.rs`) runs the mode's
  harness on the CPU replay. WikiText-2's test split is 299,078 tokens under the Qwen3 tokenizer (`tokenizers`, the
  checkpoint's `tokenizer.json`), so 73 windows of 4,096 and 36 of 8,192. Windows 61-72 at 4,096 and 7-12 at 8,192 hold
  tokens 245,760-294,911 and 49,152-98,303; the grouped control of 2026-11-01 scored tokens 0-49,151 only.
- *Computed*: the 50 rows of the pilot's sample (`gsm8k::select`, seed `0x65736d386b`) are the same in run 0 and in
  the census of run 1 of 2026-09-26, field for field once the timings are removed, so a 50-problem replay can be held
  to a census's rows at the same indices.

## 3. The condition, read in the job

The job reads r2's two sweep logs on the mount before anything else. It runs the steps of §4 only if:

1. r2's part exited 0 (the lot driver), and both logs hold their `served config:` line naming their form, the five
   lengths of the sweep, and, for the kernel arm, its load-time identity line;
2. the lead's stop line stays silent. The lead (`attention.md` §4, row D1, and §5, step V4b) stops on "in-situ 8k range
   entirely below 1.00 or 5-token median < 0.97, after one revision". `c8a01d3` is that one revision. Read on r2:
   - 5 tokens: median kernel tok/s over median grouped tok/s, at least 0.97;
   - 8,192: the upper end of the in-situ range, max kernel over min grouped, at least 1.00.

The two arms run in sequence, not round by round, so the 5-token figure is a quotient of medians, never of minima, and
it uses the tok/s the sweep prints to one decimal (*computed*). The lead's pass line for V6, 8k low end at least 1.15
and 5 tokens at least 0.99, is stricter. The operator's order of 2026-11-04 makes the stop line the trigger. A run
between the two lines is one the lead would have sent to another revision, and the report says so.

If the condition fails, the job prints the two readings and ends: outcome C0.

## 4. What runs

One image, built from the base commit. One l40sx1, inside the lot job. The sealed 4B
(`sealed-4b-2026-09-23/qwen3-4b-sealed.bin`, sha256 `886391a8…`); configs `qwen3-4b-tetra-e4-grouped.json`
(`2455fe85…`) and the kernel config (`d74ea29a…`), both written by the job and checked by sha256. Every step has its
own ceiling.

1. **Oracle** on Qwen3-0.6B, rule 10.
2. **Perplexity at 4,096**, `env LLVQ_FUSED_LAYOUT=tetra48 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0 LLVQ_KV=f16 LLVQ_EMBED=q4
   LLVQ_GQA_PPL=4096,12,60,kernel fusedrun <file>`: 12 windows from index 60, kernel against grouped, decode path in
   lockstep, the prefill path checked bit for bit on window 1. Ceiling 30 min.
3. **Perplexity at 8,192**, the same flags with `LLVQ_GQA_PPL=8192,6,6,kernel`: 6 windows from index 6, no prefill
   window. Ceiling 35 min.
4. **The band.** Both decode intervals inside [−0.12 %, +0.12 %], each end included: the job goes on. Otherwise it
   ends here (outcome K2), as the lead runs V7 only after V6 passes.
5. **Grouped replay**: `LLVQ_CONFIG=<grouped> gsm8k <file> cuda 50`, the pilot's 50 problems. It ties this image to
   the grouped census of 2026-11-04 (job `6aeb1041`, image `f77f803e`). Ceiling 10 min.
6. **GSM8K under the kernel config**, 1,319 problems, zero-shot, no reasoning block, cap 1,024 new tokens, dataset
   revision `740312ad…`: every setting of the grouped census but the config and the image. Ceiling 80 min.
7. **After the job, on the Mac, $0**: `gsm8kpair <kernel dump> <grouped census>`
   (`docs/mesures/gsm8k-grouped-2026-11-04-brut/gsm8k-4b-grouped.jsonl`), so that A − B reads kernel − grouped. The
   pair against the expanded census of 2026-09-26 is reported beside it and decides nothing.

## 5. Gates, each one voids the run

1. The image builds, and `oracle` prints MATCH.
2. The object and both configs at their bytes and sha256 on the mount.
3. Each perplexity log prints `tile 64 (served: measured optimum for sm_89)`, `int4: tv_q4_hh equals tv_q4_h bit for
   bit`, `cudarc event tracking: off (one stream, ordered by issue)`, `attn_dec.cu: embedded, sha256 c7b0d8d3…`, the
   identity line `attn_dec_g4_u64: the card equals the CPU replay bit for bit on <n> calls`, a header naming its
   window count, size and first index against `grow(256)+grouped`, one line a window, the `kernel / grouped − 1`
   summary over that count and the top-1 agreement line.
4. The 4,096 log prints `window   1: prefill path bit for bit between the two arms` exactly once (the mode stops on a
   difference, which is a wiring defect).
5. The replay ends on `"end":true` and on the pilot's fingerprint `94c7b98876288901`, and its 50 rows digest to
   `f11a0b2386062f6a1b306784d6ddd7247e0da635fc0d77e921a759610cde2f01`, the digest of the grouped census rows at the
   same indices (*computed*). The rule: drop the header and the trailer, drop `prefill_s` and `decode_s` from each row,
   write each row with sorted keys, join by newlines, sha256 of the UTF-8 bytes. On a failure the job stops before the
   census.
6. The kernel census log prints `attention kernel on grow(256)` on its `served config:` line, the tile, the int4
   parity, the kernel's sha256 and identity lines, and its dump ends on `"end":true` and on the census fingerprint
   `bfa9135ce4b55c75`.
7. `gsm8kpair` re-grades 0 rows of either dump.
8. No `LLVQ_*` variable in the job environment; each command sets its own.

A step that reaches its ceiling fails its gates.

## 6. Outcomes reported, never gated

- The two r2 readings of §3, whatever they say.
- Per size: both perplexities, `kernel / grouped − 1` with its interval, top-1 agreement, max |dlogit|, seconds.
- GSM8K: the kernel's accuracy, stops at the cap, mean tokens, decode ms a token; Δ = kernel − grouped with its 95 %
  interval, discordant counts both ways and McNemar's exact p; the same against the expanded census, reported only.
- Raw ratio = same-head ratio: every arm reads the same int4 head (rule 4).

## 7. Signed predictions

| quantity | point | interval |
|---|---|---|
| condition of §3 | **not met** | |
| if met: 5-token medians kernel / grouped | **0.98** | [0.97, 1.00] |
| 4,096: kernel / grouped − 1, decode path | **+0.000 %** | [−0.03, +0.03] |
| 4,096: half-width of its interval | **0.015 pp** | [0.005, 0.04] |
| 8,192: kernel / grouped − 1, decode path | **+0.000 %** | [−0.04, +0.04] |
| top-1 agreement, both sizes | **99.8 %** | [99.6, 99.95] |
| GSM8K replay: rows identical to the grouped census | **50 of 50** | |
| GSM8K accuracy, kernel | **82.0 %** | [80.5, 83.5] |
| Δ kernel − grouped | **+0.0 pp** | [−1.0, +1.0] |
| discordant problems, both directions | **13** | [5, 40] |
| McNemar exact p | **≥ 0.05** | |
| decode ms a token at GSM8K lengths | **7.6** | [7.2, 8.0] |

Reasoning, *estimated* unless marked. The condition: r2's own prediction at 5 tokens is 0.94 [0.90, 1.00], and the
first run read 0.931 (*measured*), so the stop line most likely fires. The perplexity: the lead's D6 puts the
decode-path change at 0.00 % ± 0.03 %. The grouped form, which moves rounding only, read +0.0095 % [−0.0048 ; +0.0238]
at 4B with a top-1 agreement of 99.788 % (*measured*, `gqa-ppl-2026-11-01.txt`); an f32 online softmax sits closer to
exact arithmetic than the f16 one it replaces, so its differences should be of that order. GSM8K: grouped against
expanded parted on 13 problems of 1,319 (*measured*, `gsm8k-grouped-2026-11-04.txt`). The step: the grouped form decodes
GSM8K at 7.46 ms (*measured*); a kernel that passed the condition is within 3 % of it at 5 tokens and gains with the
context, so about 7.6 ms at a mean position near 250.

## 8. Decision rules

- **C0.** The condition of §3 fails. Nothing runs; the two readings are reported. The kernel's T8 is not proposed on
  this lot, and the stop line goes to the operator through r2's rule 3.
- **K1.** Every gate passes, both decode intervals lie inside [−0.12 %, +0.12 %], and GSM8K reads |Δ| ≤ 1.5 pp with
  McNemar exact p ≥ 0.05 (the criterion of the engine gate of 2026-09-26 and of the grouped census). The kernel's T8 at
  4B is proposed to the operator: a new served config `configs/qwen3-4b-tetra-e4-kernel.json` (lead §6, step 5). The
  operator decides. Nothing served changes on this run.
- **K2.** Every gate that ran passes, and either a decode interval is not inside the band (GSM8K is then not run), or
  GSM8K reads |Δ| > 1.5 pp or p < 0.05. The kernel's T8 is not proposed. The report says which: an interval entirely
  outside the band, the kernel moves perplexity; an interval across a bound, a question of power, for the operator to
  order more windows or not; a GSM8K difference, its sign and size.
- **Otherwise the run is void**: a gate failed, a step reached its ceiling, or the replay of gate 5 differed (the job
  stops before the census). The T8 question is not settled. A retry goes under this prereg with its deviation beside
  it, after a new go. A kill is the operator's decision.

## 9. Cost

*Estimated* on l40sx1 at $1.80/h, inside the lot job, whose container start is paid once.

- Condition: seconds.
- Oracle, two loads: about 2 min.
- Perplexity at 4,096: about 21 min. The grouped control paired two arms over 12 windows of 4,096 in 1,482 s, 30.2 ms
  a paired step, one arm the slower expanded form (*measured*). Read as the two arms' served steps plus about 9 ms of
  harness (*computed*), kernel and grouped at the r2 rates give about 26 ms a paired step.
- Perplexity at 8,192: about 23 min. The same model over 6 × 8,191 paired steps, about 28 ms each.
- Replay: about 3 min, run 0's 160 s at today's rates.
- Kernel census: about 60 min. 1,319 × 316.9 tokens at 7.6 ms is 53 min of decode (*computed* on an *estimated*
  step), plus 7.5 min of prefill at 3.854 ms a prompt token over 88.3 tokens a problem (*measured*, the grouped census).

In all: seconds under C0; about 46 min, **$1.38**, under K2 at the band; about 109 min, **$3.27**, in full. The step
ceilings, 30 + 35 + 10 + 80 min, and the part's ceiling of 160 min, **$4.80** (*computed*). The lot job's own timeout
(180 min) can cut the census only if the parts before it overrun their estimates by more than about 30 min together; the
dump is flushed a problem at a time, so what was paid for is kept, and the run is then void by gate 6.

## 10. What this run does not establish

- The kernel at 8B or 14B, its third arm with an int8 cache, or any card but the L40S.
- The prefill path through the kernel: no call of more than 8 rows reaches it.
- MMLU under the kernel config, and GSM8K at long context.
- Whether the speed holds through the served door at every length: that is r2's sweep, read before this run.

## 11. Deviations

Any departure from this file goes in `preregistration-attn-t8-2026-11-04-ECARTS.md`, beside it, never inside it.
