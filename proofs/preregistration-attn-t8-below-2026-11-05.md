# Preregistration. The T8 extension of the attention kernel form below its threshold, on the sealed 4B: paired decode-path perplexity with every decode row on the kernel

**Written on 2026-11-05. NOT STAMPED, NOT LAUNCHED.** It is stamped with OpenTimestamps before the first measurement,
and renamed to the day of the stamp if that day is later. After the stamp it is never edited; a deviation goes beside
it in `proofs/preregistration-attn-t8-below-2026-11-05-ECARTS.md`.

Operator go: "Oui, prépare-le" (2026-11-05), for preparing this control; the launch waits for its own go in chat, with
the cost below. Cost: **$0.84 expected, $1.50 at worst** on l40sx1 (*estimated*, §8). Registry before the run: 260
rows, $270.33 (*computed*, `docs/data/jobs.csv` on `integration/lot1`).

**Base commit:** the commit of `integration/lot1` that carries this file and the `kernel0` arm of `LLVQ_GQA_PPL`; the
launcher records it. **Image:** the Space sha built from that commit, passed as `IMAGE_SHA`;
`ops/jobs/attn-t8-below.sh` refuses to launch if the Space is elsewhere, and refuses without this file and its `.ots`.

## 1. Why this run exists

- 4E v2 decodes the sealed 4B at 178.7 tok/s after a 5-token prompt against 132.1 for the eager path, with the same 64
  tokens (*measured*, `docs/mesures/pers-4e-v2-2026-11-05.txt`). At that context it runs the measurement form: every
  decode row on the split-KV kernel.
- The served config (`configs/qwen3-4b-tetra-e4-kernel-from1280.json`, operator T8 of 2026-11-05) sends a decode row
  to the kernel only from position 1,280 (`llvq_llm::kvq::KERNEL_FROM`); below, the grouped path, which uses cuBLAS
  matmuls whose rounding 4E cannot reproduce. So 4E can serve only from 1,280, and serving it at a short prompt needs
  the kernel form licensed below 1,280.
- The kernel is not bit for bit with the grouped form on the rows it takes. Its T8 from 1,280 was read on those
  positions only (`docs/mesures/attn-t8-r2-2026-11-04.txt`: −0.0013 % [−0.0074 ; +0.0047] at 4,096 and +0.0001 %
  [−0.0076 ; +0.0077] at 8,192). No position below 1,280 has ever been scored under the kernel. This run scores them.

## 2. What the code holds

- `LLVQ_GQA_PPL=<ctx>,<windows>,<first>,kernel0` (`bin/fusedrun`), new in the base commit: the `kernel` mode with the
  model's kernel threshold set to 0 by `Qwen3::set_kernel_from_for_measurement`, the call 4E's eager reference already
  uses. It pairs `grow(256)+grouped` with `grow(256)+kernel` window by window through `Qwen3::decode_nll_split` with
  split 0: no position is checked bit for bit, every one is scored, and the `from 0` summary is the whole decode path.
  It prints `measurement form: every decode row on the kernel, from position 0 (the served config runs the grouped
  path below 1280)`. The prefill path stays grouped in both arms (a call of more than `max_rows` rows never reaches
  the kernel, `model.rs` around line 1843), so window 1's prefill is checked bit for bit as in the `kernel` mode.
- Checked on the Mac at $0: the parser test (`gqa_ppl_spec`) accepts `1280,24,40,kernel0`, refuses `kernel0` without
  its first window, and still refuses `1280,24,40,kernel` (never reaches the kernel). The Linux half type-checks in
  `ops/check-cuda.sh`, and a deliberate type error in the new lines is caught there. Clippy: the two existing
  `attn_cuda.rs` warnings only.
- The kernel text is `llvq-cuda/kernels/attn_dec.cu`, sha256 `c7b0d8d3…`, the text of attn-t8-r2 and 4E v2.
- No config of `configs/` is read or written. The flags are the served ones, spelled out:
  `LLVQ_FUSED_LAYOUT=tetra48 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0 LLVQ_KV=f16 LLVQ_EMBED=q4`.

## 3. What runs

One image, built from the base commit. One job, l40sx1, `ops/jobs/attn-t8-below.sh`, timeout 50 min. The sealed 4B
(`sealed-4b-2026-09-23/qwen3-4b-sealed.bin`, sha256 `886391a8…`). In order, each step under its own ceiling:

1. **Oracle** on Qwen3-0.6B, rule 10. Ceiling 5 min.
2. **Below the threshold:** `LLVQ_GQA_PPL=1280,32,77,kernel0`, 32 windows of 1,280 (indices 77 to 108, tokens 98,560
   to 139,519, scored by no earlier perplexity run). Every scored position lies below 1,280: 32 × 1,279 = 40,928
   positions. Ceiling 18 min. If its interval leaves the band, the job ends (N1).
3. **The whole object under the extended form:** `LLVQ_GQA_PPL=4096,12,60,kernel0`, 12 windows of 4,096 (indices 60
   to 71, tokens 245,760 to 294,911), attn-t8-r2's windows. 12 × 4,095 = 49,140 positions, 15,360 of them below 1,280.
   Ceiling 24 min.

## 4. Gates, each one voids the run

A job that exits non-zero, or ends without an `OUTCOME` line in `outcome.txt`, is void.

1. The image builds, and `oracle` prints MATCH.
2. The object at its bytes and sha256 on the mount.
3. Each perplexity log prints `tile 64 (served: measured optimum for sm_89)`, `int4: tv_q4_hh equals tv_q4_h bit for
   bit`, `cudarc event tracking: off (one stream, ordered by issue)`, `attn_dec.cu: embedded, sha256 c7b0d8d3…`, the
   identity line `attn_dec_g4_u64: the card equals the CPU replay bit for bit on <n> calls`, and the measurement-form
   line of §2.
4. Each perplexity log prints its header naming its window count, size and first index, `against grow(256)+grouped,
   decode path in lockstep` and `prefill path on 1 window(s), checked bit for bit`; the line `the kernel form takes
   positions 0 and later`; one window line per window marked `every position on the kernel`; the line `window   1:
   prefill path bit for bit between the two arms` exactly once; the paired summary; both top-1 lines.
5. No `LLVQ_*` variable in the job environment; each command sets its own.

A step that reaches its ceiling fails its gates.

## 5. Outcomes reported, never gated

- Per size: both arms' decode ppl, the paired interval of `kernel / grouped − 1`, top-1 agreement, max |dlogit|,
  seconds.
- Raw ratio = same-head ratio: both arms read the same int4 head (rule 4).

## 6. Signed predictions

| quantity | point | interval |
|---|---|---|
| 1,280 × 32: kernel / grouped − 1 | **+0.000 %** | [−0.03, +0.03] |
| 1,280 × 32: half-width of the interval | **0.010 pp** | [0.004, 0.03] |
| 4,096 × 12: kernel / grouped − 1 | **+0.000 %** | [−0.03, +0.03] |
| 4,096 × 12: half-width of the interval | **0.008 pp** | [0.003, 0.03] |
| top-1 agreement, both sizes | **99.8 %** | [99.6, 99.95] |
| outcome | **P** | |
| job running time | **28 min** | [20, 42] |

Reasoning, *estimated*. The kernel's f32 online softmax replaces the grouped form's f16 one and moves rounding only.
On the positions it already takes it read within ±0.008 % (attn-t8-r2), and the grouped form against the expanded one,
which also moves rounding only, read +0.0095 % [−0.0048 ; +0.0238] (`gqa-ppl-2026-11-01.txt`). Short positions attend
over few keys, where both forms are closest to exact, so no larger effect is expected below 1,280.

## 7. Decision rules

Evaluated in this order; every run lands in exactly one.

- **V (void):** a gate of §4 fails, a step reaches its ceiling, or the job ends without an `OUTCOME` line. Nothing is
  read but the cause, which is diagnosed before any rerun (rule 9).
- **N1 (not proposed):** at 1,280 × 32 the interval is not inside [−0.12 %, +0.12 %], each end included. The job ends
  there.
- **N2 (not proposed):** at 4,096 × 12 the interval is not inside the band.
- **P (proposed):** every gate holds and both intervals lie inside the band. The T8 extension of the kernel form below
  1,280 is proposed to the operator, with this run's numbers. The decision is the operator's; nothing served changes
  before it, and no file of `configs/` is written by this run. Serving 4E would then also need a served config that
  names it and its own served-door measurement, under their own prereg.

The band is the one of `docs/ETAT.md` §4 and of every attention T8 before this one, ±0.12 % of paired perplexity.

## 8. Cost

*Estimated* on l40sx1 at $1.80/h.
- Oracle, setup and two loads with the 103-call kernel check: 3 min.
- Part 2: a paired step is about one grouped and one kernel decode step at a short context, 7.4 + 7.9 ms (*measured*
  rates, 135.0 and 127.1 tok/s after 5 tokens, attn-t8-r2 and lot 1): 19.6 s a window of 1,279 steps, 32 windows,
  plus the prefill check: 11 min.
- Part 3: attn-t8-r2 estimated 69.5 s a window of 4,096 for the threshold form; the kernel below 1,280 adds about
  0.5 ms on 1,279 steps: 70 s a window, 12 windows: 14 min.
- In full: about 28 min, **$0.84**. N1 ends near 14 min ($0.42).
- Worst: the timeout, 50 min, **$1.50**.

## 9. What this run does not establish

- A task score below 1,280 under the kernel form: `gsm8k` has no measurement form that moves the threshold, and MMLU
  scores one prefill forward, which never reaches the kernel. GSM8K's decodes all lie below 1,280 (the census reaches
  position 1,200 at most), so a GSM8K census under the extended form would be the task-level check, at its own cost.
- 4E's served speed through the served door: 4E has no served config yet.
- The 8B and the 14B: the kernel form has never run there.
- Any card but the L40S.
