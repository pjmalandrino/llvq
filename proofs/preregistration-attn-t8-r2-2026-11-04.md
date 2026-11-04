# Preregistration. The T8 control of the attention kernel form with its threshold, on the sealed 4B: GSM8K replay, served sweep, paired decode-path perplexity from position 1,280

**Written on 2026-11-04. NOT STAMPED, NOT LAUNCHED.** It is stamped with OpenTimestamps before the first measurement,
and renamed to the day of the stamp if that day is later. After the stamp it is never edited; a deviation goes beside
it in `proofs/preregistration-attn-t8-r2-2026-11-04-ECARTS.md`.

**It supersedes `proofs/preregistration-attn-t8-2026-11-04.md`** (sha256 `40f0ac95…`, stamped), which ran no step:
in lot 1 its condition failed (5 tokens 0.9408 < 0.97, outcome C0, `docs/mesures/lot1-4b-2026-11-04.txt`). That file
stays as it is. The object it controlled, a kernel form with no threshold, is no longer the one proposed.

Operator go: "termine tout tu as mon go pour la nuit" (2026-11-04, relayed by the session of the throughput audit), to be
confirmed in chat before the launch. Cost: **$1.62 expected, $2.40 at worst** on l40sx1 (*estimated*, §8). Registry
before the run: 252 rows, $261.96, 5 unpriced (*computed*, `docs/data/jobs.csv` on `integration/lot1`, 2026-11-04).

**Base commit:** the commit of `integration/lot1` that carries this file, after the merge of
`claude/quizzical-kowalevski-38c5fc`; the launcher records it. **Image:** the Space sha built from that commit, passed
as `IMAGE_SHA`; `ops/jobs/attn-t8-r2.sh` refuses to launch if the Space is elsewhere, and refuses without this file and
its `.ots`.

## 1. Why this run exists

- The split-KV attention kernel decodes the served 4B faster than the grouped form at long context and slower at short
  context: in r2, 104.2 against 77.7 tok/s at 8,192 (×1.341) and 127.1 against 135.1 after 5 tokens (×0.941)
  (*measured*, job 6aeb90e3, `docs/mesures/lot1-4b-2026-11-04.txt`).
- The kernel form now takes a row to the kernel only from position 1,280 (`llvq_llm::kvq::KERNEL_FROM`, commit
  `1d3bf09`); below, the grouped path, the grouped form's bits. The threshold was chosen from r2 alone: the two forms
  cross in [1,130, 1,382] positions, at 1,239 on the straight lines (*computed*,
  `docs/mesures/attn-threshold-host-2026-11-04.txt` §1).
- The kernel is not bit for bit with the grouped form on the rows it takes. Serving the kernel form needs a T8
  decision of its own (`docs/archive/debit-vague5-2026-11-02/attention.md` §3.1 and §7). This run is that control:
  the paired decode-path perplexity restricted to the positions the kernel takes, a GSM8K replay that checks the
  wiring below the threshold, and the served sweep that measures what the threshold does to the speed.

## 2. What the code holds

- `"attention": "kernel"` in a served config: rows at position p ≥ 1,280 go to the kernel, the rows below to the
  grouped path; a call that straddles 1,280 is split by row. The served line reads `attention kernel from position
  1280 on grow(256)`; an image without the threshold prints `attention kernel on grow(256)` and fails gate 3.
- `LLVQ_GQA_PPL=<ctx>,<windows>,<first>,kernel` (`bin/fusedrun`): pairs `grow(256)+grouped` with `grow(256)+kernel`
  window by window through `Qwen3::decode_nll_split`, split at 1,280. It stops on any step below 1,280 whose two logit
  vectors differ in any bit, and prints, beside the whole window's, a `from 1280` line: the paired 95 % t interval of
  `kernel / grouped − 1` over the positions 1,280 and later, the positions the kernel takes. It refuses a window that
  does not reach position 1,281.
- `LLVQ_CONTEXTS` (`bin/fusedrun`) prints below its table one `tokens after <n>: <fingerprint> <ids>` line a length,
  the 256 tokens of the reference round.
- *Measured* on the Mac, $0 (`docs/mesures/attn-threshold-host-2026-11-04.txt`): in a 2-layer Qwen3 on the CPU
  replay, positions 1,270 to 1,279 bit for bit with the grouped form and 1,280 to 1,289 each different, within 5e-3;
  the split call returns, for every row at the threshold or past it, the kernel's bits for that row alone. 14 mutants
  of the threshold and the harness killed. The fast suite passes (999 tests); clippy at zero on the Mac and for Linux.
- *Measured* on the Mac, $0: `ops/jobs/attn-t8-r2.sh` with `DRY_RUN=1` parses; its emitted job script, run on the Mac
  over fake card binaries that print the lines of r2's logs and of `fusedrun`'s format strings, ends on P, N1, N2 and
  N3 in their scenarios, and voids (non-zero exit, no outcome line) on differing tokens below the threshold, on a
  differing step below the threshold, and on an image without the threshold.
- The kernel text is `llvq-cuda/kernels/attn_dec.cu`, sha256 `c7b0d8d3…`, the text r2 measured.
- The kernel config is written by the job: the served grouped config (`2455fe85…`) with the attention field set to
  `kernel` and a note saying it is a measurement, sha256 `e01326fc…`. It is not a file of `configs/` and changes no
  served default.
- *Computed*: the 50 pilot problems (`gsm8k::select`, seed `0x65736d386b`) reach position 844 at most in the grouped
  census of 2026-11-04 (n_prompt + n_gen); the 1,319 problems reach 1,200. No GSM8K problem of the census reaches the
  kernel, so GSM8K under the kernel config can only reproduce the grouped census: the replay checks the wiring, the
  full census would test nothing about the kernel and is not run.

## 3. What runs

One image, built from the base commit. One job, l40sx1, `ops/jobs/attn-t8-r2.sh`, timeout 80 min. The sealed 4B
(`sealed-4b-2026-09-23/qwen3-4b-sealed.bin`, sha256 `886391a8…`). The flags of the perplexity steps are the served
ones: `LLVQ_FUSED_LAYOUT=tetra48 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0 LLVQ_KV=f16 LLVQ_EMBED=q4`. In this order, each step under
its own ceiling:

1. **Oracle** on Qwen3-0.6B, rule 10. Ceiling 5 min.
2. **(c) GSM8K replay** under the kernel config: `LLVQ_CONFIG=<kernel> gsm8k <file> cuda 50`, dataset revision
   `740312ad…`, the pilot's 50 problems, zero-shot, no reasoning block, cap 1,024 new tokens. Its rows are digested by
   the rule of `ops/jobs/attn-t8.sh` (header and trailer dropped, `prefill_s` and `decode_s` dropped, sorted keys,
   joined by newlines, sha256) and compared with `f11a0b2386062f6a1b306784d6ddd7247e0da635fc0d77e921a759610cde2f01`,
   the digest of the grouped census rows at the same indices. If they differ, the same 50 problems run under the
   grouped config, to say whether the image or the kernel form moved, and the job ends (N1). Ceiling 8 min each.
3. **(a) Served sweep**: `LLVQ_CONFIG=<c> LLVQ_CONTEXTS=5,256,1024,1280,2048,4096,8192 fusedrun <file> 256`, the
   kernel config then the grouped config, 1 discarded and 5 timed rounds each. Ceiling 13 min an arm.
4. **(b) Perplexity at 4,096**: `LLVQ_GQA_PPL=4096,12,60,kernel`, 12 fresh windows (indices 60 to 71, tokens
   245,760 to 294,911), the prefill path checked bit for bit on window 1. Ceiling 24 min. If its `from 1280` interval
   leaves the band, the job ends (N2).
5. **(b) Perplexity at 8,192**: `LLVQ_GQA_PPL=8192,6,6,kernel`, 6 windows (indices 6 to 11, tokens 49,152 to 98,303),
   no prefill window. Ceiling 26 min.

Positions scored by the kernel: 12 × 2,815 = 33,780 at 4,096 and 6 × 6,911 = 41,466 at 8,192 (*computed*).

## 4. Gates, each one voids the run

A job that exits non-zero, or ends without an `OUTCOME` line in `outcome.txt`, is void.

1. The image builds, and `oracle` prints MATCH.
2. The object at its bytes and sha256 on the mount; both configs written at sha256 `e01326fc…` (kernel) and
   `2455fe85…` (grouped).
3. Every log of the kernel config prints its `served config:` line with `attention kernel from position 1280 on
   grow(256)`, `tile 64 (served: measured optimum for sm_89)`, `int4: tv_q4_hh equals tv_q4_h bit for bit`,
   `attn_dec.cu: embedded, sha256 c7b0d8d3…` and the identity line `attn_dec_g4_u64: the card equals the CPU replay
   bit for bit on <n> calls`. The grouped sweep prints `attention grouped on grow(256)`, the tile and the parity.
4. The kernel replay dump ends on `"end":true` and on the pilot's fingerprint `94c7b98876288901`.
5. Each sweep prints seven table rows and seven `tokens after` lines.
6. The `tokens after` lines at 5, 256 and 1,024 are identical between the two configs: every position those decodes
   take lies below 1,280 (the last is 1,024 + 255 = 1,279).
7. Each perplexity log prints its header naming its window count, size, first index and `against
   grow(256)+grouped, decode path in lockstep`, the line `the kernel form takes positions 1280 and later`, one window
   line per window marked `positions < 1280: bit for bit`, the whole-window summary, the `from 1280` summary, both
   top-1 lines, `cudarc event tracking: off (one stream, ordered by issue)`, and at 4,096 the line `window   1: prefill
   path bit for bit between the two arms` exactly once.
8. No step below position 1,280 differs between the two arms (the mode stops on one: a wiring defect).
9. No `LLVQ_*` variable in the job environment; each command sets its own.

A step that reaches its ceiling fails its gates.

## 5. Outcomes reported, never gated

- Sweep, per length: both tok/s with their ranges, the quotient of medians and the in-situ range [min kernel / max
  grouped, max kernel / min grouped], and at 1,280 and above whether the 256 tokens are identical or where they first
  diverge (a free-running greedy decode at long context, informative).
- Perplexity, per size: both arms' ppl on the whole window and from 1,280, both intervals, top-1 agreement on both
  spans, max |dlogit|, seconds.
- Replay: accuracy, mean tokens, decode ms a token.
- Raw ratio = same-head ratio: every arm reads the same int4 head (rule 4).

## 6. Signed predictions

| quantity | point | interval |
|---|---|---|
| kernel replay rows equal to the grouped census rows | **50 of 50** | |
| tokens identical at 5, 256, 1,024 | **yes** | |
| sweep, kernel / grouped medians at 5 | **1.000** | [0.98, 1.02] |
| at 256 | **1.000** | [0.98, 1.02] |
| at 1,024 | **1.000** | [0.98, 1.02] |
| at 1,280 | **1.01** | [0.97, 1.05] |
| at 2,048 | **1.06** | [1.02, 1.10] |
| at 4,096 | **1.20** | [1.15, 1.25] |
| at 8,192 | **1.34** | [1.28, 1.40] |
| grouped arm tok/s against r2 (135.1 / 132.0 / 114.6 / 98.6 / 77.7) | **same** | each within ±2 % |
| lengths 1,280 to 8,192 whose 256 tokens are identical between the configs | **2 of 4** | [0, 4] |
| 4,096, from 1,280: kernel / grouped − 1 | **+0.000 %** | [−0.03, +0.03] |
| 4,096, from 1,280: half-width of the interval | **0.015 pp** | [0.005, 0.04] |
| 8,192, from 1,280: kernel / grouped − 1 | **+0.000 %** | [−0.04, +0.04] |
| 8,192, from 1,280: half-width of the interval | **0.02 pp** | [0.005, 0.05] |
| top-1 agreement from 1,280, both sizes | **99.8 %** | [99.6, 99.95] |
| outcome | **P** | |
| job running time | **54 min** | [45, 72] |

Reasoning, *estimated* unless marked.
- Replay and tokens: no position of these decodes reaches 1,280 (*computed*, §2), and below 1,280 the kernel form runs
  the grouped form's code with the same arguments (host-proven, §2).
- Sweep: at 5, 256 and 1,024 the two configs run the same arithmetic, so the ratio is the run-to-run drift of two arms
  run in sequence (r2's grouped arm sat within 1 % of the first run's). At 1,280 the decode spans positions 1,280 to
  1,535: the grouped line gives 8.32 ms, the kernel 8.22 (interpolated between its 256 and 2,048 medians). From 2,048
  on, the decode is all kernel, as in r2.
- Tokens past 1,280: a top-1 flip rate near 0.2 % a step over 256 steps leaves about a 60 % chance of no flip per length.
- Perplexity: the grouped form against the expanded one, which moves rounding only, read +0.0095 % [−0.0048 ;
  +0.0238] at 4B with a top-1 agreement of 99.788 % (*measured*, `gqa-ppl-2026-11-01.txt`). The kernel's f32 online
  softmax sits closer to exact arithmetic than the f16 one it replaces, so its difference should be of that order.
  Restricting the score to the positions the kernel takes removes the identical part, which would only dilute it.

## 7. Decision rules

Evaluated in this order; every run lands in exactly one.

- **V (void):** a gate of §4 fails, a step reaches its ceiling, or the job ends without an `OUTCOME` line. Nothing is
  read but the cause, which is diagnosed before any rerun (rule 9).
- **N1 (not proposed):** the kernel replay's rows do not digest to `f11a0b23…`. The grouped replay says whether the
  image (grouped differs too) or the kernel form below its threshold (grouped equal) moved. The job ends there.
- **N2 (not proposed):** at 4,096, the `from 1280` interval is not inside [−0.12 %, +0.12 %], each end included. The
  job ends there.
- **N3 (not proposed):** at 8,192, the `from 1280` interval is not inside the band.
- **P (proposed):** every gate holds, the replay digest is equal, and both `from 1280` intervals lie inside the band.
  The T8 of the kernel form with its threshold at 1,280 is proposed to the operator, with this run's numbers. The
  decision is the operator's; nothing served changes before it, and no file of `configs/` is written by this run.

Readings that move no outcome:
- If the in-situ range at 1,280 lies entirely below 1.00, the threshold is too low by this measurement: a proposal to
  raise it goes to the operator, with the sweep. Entirely above 1.00 at 1,024 cannot be read here (both configs run
  the grouped path there).
- The whole-window intervals are reported beside the `from 1280` ones and decide nothing.

The stop line of the superseded prereg (5 tokens at least 0.97) is not used: the kernel form now decodes a short
context on the grouped path, so that line would hold by construction and test nothing about the kernel.

## 8. Cost

*Estimated* on l40sx1 at $1.80/h.
- Oracle, setup and loads: 2 min.
- (c): the 50 problems took 133.5 s of prefill and decode in the grouped census (*measured*, sum of `prefill_s` and
  `decode_s`), plus the load and the 103-call check: 2.5 min.
- (a): r2's sweep took 7 min 20 s an arm for five lengths (*measured*, part log of lot 1); 1,024 and 1,280 add 6 rounds
  of a 4 s and a 5 s prefill and a 2.1 s decode: 8.7 min an arm, 17.4 min.
- (b): a paired step costs about the two forms' served decode at that position. The grouped control of 2026-11-01
  (*measured*, 123.5 s a window of 4,096 with two prefills of about 17 s) leaves 89.6 s for 4,095 steps, which is the
  sum of the expanded and grouped served costs over those positions (92.1 s) within 3 %. Here, 8.77 ms (grouped) + 8.17 ms (kernel
  form) a step at 4,096: 69.5 s a window, 12 windows, the prefill check and the load, 14.8 min. At 8,192, 10.14 +
  8.73 ms: 155 s a window, 6 windows, 15.8 min.
- In full: about 53 min, **$1.62** with a little slack. N1 ends near 7 min ($0.21); N2 near 37 min ($1.11).
- Worst: the timeout, 80 min, **$2.40**.

## 9. What this run does not establish

- Speed against the grouped form round by round: the sweep's arms run in sequence; its ratios are in situ, not paired.
- The crossover below 1,280: neither config runs the kernel there.
- Quality on a free-running task past 1,280. The decode-path perplexity is teacher-forced; the sweep's token lines
  are the only free-running reading past the threshold, on a repeated-seed prompt, and decide nothing.
- The 8B and the 14B; any card but the L40S.

## 10. Deviations

Any departure after the stamp goes in `proofs/preregistration-attn-t8-r2-2026-11-04-ECARTS.md`.
