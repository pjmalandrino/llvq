# Preregistration. Three bit-exact gestures at the token boundary of the decode loop, one mechanism an arm, with direct probes and the host clocks (2026-11-04)

**Written on 2026-11-04. NOT STAMPED, NOT LAUNCHED.** It is stamped with OpenTimestamps before the first measurement,
and renamed to the day of the stamp if that day is later. Operator go: "ok pour les 3" on 2026-11-04, in the audit
session, for this run, the attention kernel and the 32B walls, with a worst case of $0.60 for this one. Cost: about
**$0.35** on l40sx1 (*estimated*, §8); timeout 20 min, so **$0.60** at worst (*computed*). Cap: $0.60, the operator's. Registry before it: $261.35 over 251 rows (*measured*, `docs/data/jobs.csv`, 2026-11-04).

**Base commit:** `02ec809323eb`, branch `integration/lot1`, which merges the code of §2 (`e3a6fde`, branch tip
`ccbb3c3`) with the attention branch (tip `5255470`) and the 32B wall branch (tip `78285a5`); this file is committed on
top of it. **Space sha:** `142216fc72326a98b94316a7f6cd2e989f0efc50`, built from `02ec809`. The job script is
`ops/jobs/gesture-ab.sh`; it refuses to launch without this file and its `.ots`. It runs as the first part of the lot
job `lot1-4b` (`ops/jobs/lot1.sh`), §3.

The merge adds to the image, beside §2's code: the attention form `kernel` (a `Block` branch taken only under that form;
the expanded and grouped dispatch unchanged), the bench `attnbench`, the `LLVQ_GQA_PPL` kernel mode, the wide rotation
`rot_apply_dev` and the f16-staged int4 matvec `tv_q4_hs` in units of their own, and the int4 load path's per-entry
sizes in `fused_cuda.rs` (`git diff e3a6fde 02ec809`). None is selected by this run's flags or config. Gates 4 and 5 are
what checks that the default path did not move under them.

Draft: `docs/archive/debit-vague5-2026-11-02/prereg-gestures-1.md`. §11 lists what changed from it.

## 1. Why this run exists

The 4B decodes at 133.3 tok/s through the served door under the grouped config, 7.50 ms a token (*measured*,
`docs/mesures/served-grouped-2026-11-02.txt`). Three cheap levers at the token boundary have never been measured.

- **The cast.** Each token casts the 151,936 f16 logits to f32 before candle's argmax (`model.rs`, `generate`).
  candle's `fast_argmax_f16` and `fast_argmax_f32` are one template, and widening f16 to f32 is exact and monotone, so
  the cast changes no token. It costs one launch and one 608 KB allocation a token, about 2-4 µs (*estimated*).
- **candle's argmax.** It runs 151,936 iterations on one block of 1,024 threads, each with a 64-bit `%` and `/`
  (`cuda_utils.cuh:25-38`), and uploads its dims and strides from pageable memory on every call
  (`cuda_backend/mod.rs:333`): 15-50 µs (*estimated* from the code). A key kernel over 75 blocks reproduces its tie
  order: 4-6 µs (*estimated*).
- **The wait.** Each token reads the id back to the host and uploads it again. Fenced, that boundary costs 0.078-0.101 ms
  (*measured*, `phases-2026-08-07.txt`). Kept on the card, the host stops waiting and the device stops starving at
  each token: a bubble of 20-45 µs (*estimated*).

No journal holds the host span of the served step either. Two facts suggest the host is hidden behind the device, a
reading that is *estimated* and rests on two assumptions (`docs/archive/debit-vague5-2026-11-02/gestures.md` §3.2). The
clocks of §3 decide it, read on the device-token step, where nothing waits on the device.

The int4 `v_proj` on a side stream (Part C of the draft) is out of this run: it is measured only on the commit that
carries the operator's norm and rotation fusion, and only under a ruling on cudarc's events with two streams.

## 2. What the code holds

All of it off by default: `generate` and every published path run as before.

- `Qwen3::logits_last_native` and `Qwen3::generate_loop` (`llvq-llm/src/model.rs`), with four kinds beside `generate`,
  each one mechanism from its neighbour (`TokenLoop`): `F16` (no cast), `Key` (the key kernel into a one-slot ring,
  `to_scalar` of its rank-0 view, `from_slice`), `OnDevKey` (no per-token readback or upload; the next step embeds a
  view of `ring[pos]`; one readback at the end), `OnDevCandle` (candle's argmax output embedded as the next input).
- `tok_argmax_h` (`llvq-llm/kernels/tok_argmax.cu`), its own NVRTC module on candle's stream: a 64-bit key that
  reproduces candle 0.9.2's tie order (earliest index inside a thread, smallest `bitrev10` across threads), 75 blocks
  of 256 threads, written into the ring by a candle `InplaceOp2` (`llvq-llm/src/gesture_cuda.rs`). A NaN logit raises
  a sticky flag and writes token 0; the loop then refuses the generation.
- The host reference of the key, its case list and its card mutants (`llvq-llm/src/argmax_key.rs`).
- `LLVQ_GESTURE_AB=1` in `bin/fusedrun`: refused beside `LLVQ_CONFIG`, beside every other measurement mode, with
  `LLVQ_KV_PREALLOC` and with `LLVQ_KERNEL_DIR`. No new binary, so no change to `ops/Dockerfile.cuda` or `UPLOAD_ALLOW`.

*Measured* on the Mac, $0:
- Every kind returns `generate`'s tokens on `Cat`, `Grow(4)` and `Prealloc(32)`, under the expanded, shared and grouped
  forms, in f32 and f16, with groups of 2, 4 and 5 (`token_loops_match_generate`). The key kinds hand the key, at each
  of 9 steps, the logits of the step API bit for bit (`key_loops_feed_the_reference_logits`).
- The key returns the replayed candle index on 8 adversarial cases, 32 sparse tie sets in the debug loop and 300 in the
  release loop, and one dense set; at least 28 of the 32 sparse sets tie across threads and resolve outside thread 0.
- 14 host mutants, each killed by a test: 8 in the loop (a ring view at slot 0, a frozen offset twice, a write one slot
  late, a break before the write, the first candle pick fed again, the prompt's first position, a slot read before its
  write) and 6 in the key and its replay (no −0 mapping, thread order by index, latest index in a thread, a 9-bit
  reversal, the NaN ignored, the replay keeping the right operand on ties).
- `cargo clippy --all-targets` at zero warnings on the Mac; the Linux half (`llvq-llm --features cuda`) at zero warnings
  in the `llvq-check` image, with a planted type error in `gesture_cuda.rs` caught first.
- NVRTC 12.4 compiles `tok_argmax.cu`, its nine mutants and `gesture_spin.cu` for `compute_89` and `compute_80` with an
  empty log, in the `llvq-check` image.

Not established: the kernel has never run on a card. The load check and the card mutants first run in this job.

## 3. What runs

One image built from the base commit. One job, l40sx1: the lot job `lot1-4b`, in which this run is the first of four
parts, each its own bash process under its own ceiling (18 min for this one), the 32B wall check, the attention bench
r2 and the attention T8 control after it. The job script below is the one `ops/jobs/gesture-ab.sh` writes, run as it
stands. It reads the sealed 4B (`sealed-4b-2026-09-23/qwen3-4b-sealed.bin`,
sha256 `886391a8…`), the config `qwen3-4b-tetra-e4-grouped.json` (`2455fe85…`), written by the job from its launch
command and checked by sha256.

1. **Oracle** on Qwen3-0.6B, rule 10.
2. **The bench door, replayed.** `fusedrun <file> 256` with the served flags spelled out and `LLVQ_EMBED=q4`, against
   the dense arm, as in run 1 of the paper table. It checks that the default path did not move.
3. **The served door, grouped, unchanged.** `LLVQ_CONFIG=<grouped> fusedrun <file> 256`. The gestures are not served by
   this commit, so the served text must not move. It is extracted as `ops/jobs/served-grouped.sh` extracted it, from
   the `served : ` line to `No dense arm on this path`.
4. **The A/B.** The served flags plus `LLVQ_GESTURE_AB=1`, 256 new tokens after the 5-token prompt. One model, put in the
   served 4B state by hand (`Grow(256)` and grouped, as `LLVQ_GQA_AB` does), cudarc's event tracking turned off on one
   stream before the load, as the served door does. In order, each step gating the next:
   - P1, the load check: 41 calls in one sequence through one slot, ticket and flag, each against candle's own `argmax`
     on the same device tensor and against the host replay: 8 adversarial cases, the first 32 sparse tie sets, the NaN
     case last, with the flag read before and after it.
   - P2, the card mutants m1 to m9, each applied in memory to the embedded text with every edit found exactly once,
     compiled as its own NVRTC module, run through the same sequence; each must fail it.
   - P3, the kernel-time probe, on the f16 logits of the prompt's last position, 5 × 100 back-to-back launches each,
     event-timed, median: candle's `fast_argmax_f16` and `fast_argmax_f32` launched directly from candle's module with
     their info buffer uploaded once; the logits cast through candle; `tok_argmax_h`; and candle's full `argmax` call.
     The direct launch, the full call and the key must name the same token.
   - P4, the pageable-copy probe, five times each behind a 5 ms spin on candle's stream: the host time of
     `Tensor::from_slice` of one id, of candle's `argmax` on the logits row, and of `to_scalar` on one u32 (the control,
     which must wait).
   - The arms, every arm in every round, 2 rounds discarded and 5 timed: `Base` (`generate`), `F16`, `Key`, `OnDevKey`,
     `OnDevCandle`.
   - The clocks. `E_base` and `E_ondev`: `(T(64) − T(1)) / 63` for the `Base` and `OnDevKey` arms, five interleaved
     passes, each pass ending on its readback. Then the spun `OnDevKey` loop, 5 passes of 63 decode steps, six segments
     a step, each behind a 5 ms spin (the step start, then after ln1 of layers 6, 12, 18, 24 and 30 through the capture
     hook), giving H, the host enqueue span, and D, the device time. A step is valid only if every segment was enqueued
     at least 1 ms before its spin could end.

Every arm includes the 5-token prefill in its clock, as the trims A/B did: about 23 ms of about 1.9 s (*computed*), so
every ratio is diluted toward 1 by about 1 %, and that is declared here.

## 4. Gates, each one voids the run

1. The image builds, and `oracle` prints MATCH.
2. The sealed file and the config at their bytes and sha256 on the mount.
3. Every load prints `int4: tv_q4_hh equals tv_q4_h bit for bit` and `tile 64 (served: measured optimum for sm_89)`;
   no `LLVQ_*` variable is in the job environment.
4. Step 2 prints `fused F0: 256 tokens identical to the dense arm`, exactly one `  fused F0:  ` line, and that line
   hashes to `8784ffc423bfbeeaeb9a92abd426573cc45f9a083ba56e61e52d9ff40e3b018f`.
5. Step 3 prints `attention grouped on grow(256)` on its `served config: ` line, `No dense arm on this path`, exactly one
   `served : ` line, and its extracted decoded text hashes to
   `0597cee8b20befecdf92c2f3f2f1938d8216bd244542faab42d2637920a3c12a`, the digest of the text the served door decoded
   under both configs on 2026-11-02 (*measured*, `docs/mesures/served-grouped-2026-11-02-brut/lines.sha256`). The digest
   is compared with a text taken by the same door and the same extraction: the bench door's line of gate 4 is printed in
   another format and is never compared with it.
6. Step 4 prints, in this order: `cudarc event tracking: off (one stream, ordered by issue), before the load`;
   `attention grouped on grow(256), layout tetra48`; `P1 tok_argmax: candle's fast_argmax_f16 index on 40 cases in
   sequence, NaN flag raised`; `P2 9 of 9 card mutants killed on the card, each edit found exactly once` and one line a
   mutant; the P3 and P4 lines; `5 arms × 5 interleaved rounds after 2 discarded, 256 identical tokens in every arm and
   round`, the arm table and five ratio lines; the clock E line; the clock H/D line over 315 spun steps. The mode stops
   before any number if an arm in any round, or a clock pass, returns a token that differs from `Base`.

## 5. Outcomes reported, never gated

- Each arm's tok/s, median of 5 rounds with its range; the four single-mechanism ratios round by round: `F16/Base`
  (a1), `Key/F16` (a2), `OnDevKey/Key` and `OnDevCandle/F16` (a3); `OnDevKey/Base`, labelled as three mechanisms.
- P3: µs a launch for each of the five; P4: the three host times in ms, with the control.
- `E_base`, `E_ondev`, H and D in ms a step, slack = D − H, `X_ondev` = `E_ondev` − D, `E_base` − `E_ondev`, and the
  count of valid spun steps.
- Raw ratio = same-head ratio: every arm reads the same int4 head (rule 4).

## 6. Signed predictions

| quantity | point | interval |
|---|---|---|
| step 2, fused tok/s (bench door) | **121** | [115, 126] |
| step 3, served tok/s, grouped | **133** | [128, 138] |
| `Base` tok/s | **133** | [128, 138] |
| `F16 / Base` | **1.000** | [0.998, 1.003] |
| `Key / F16` | **1.004** | [1.000, 1.008] |
| `OnDevKey / Key` | **1.004** | [1.001, 1.008] |
| `OnDevCandle / F16` | **1.003** | [0.998, 1.008] |
| `OnDevKey / Base`, reported | **1.008** | [1.002, 1.015] |
| P3 candle `fast_argmax_f16`, direct, µs | **25** | [8, 60] |
| P3 candle `fast_argmax_f32`, direct, µs | **25** | [8, 60] |
| P3 logits cast, µs | **3** | [2, 6] |
| P3 `tok_argmax_h`, µs | **5** | [3, 9] |
| P3 candle `argmax` call, µs | **28** | [10, 70] |
| P4 `from_slice` behind the spin, ms | **< 0.5** (no wait) | |
| P4 candle `argmax` behind the spin, ms | **< 0.5** (no wait) | |
| P4 `to_scalar` control, ms | **5.0** | [4.5, 5.6] |
| `E_base`, ms a decode step | **7.44** | [7.20, 7.70] |
| `E_ondev`, ms a decode step | **7.37** | [7.10, 7.65] |
| H, ms a step | **3.8** | [2.0, 6.5] |
| D, ms a step | **7.30** | [7.00, 7.60] |
| slack D − H, ms | **3.5** | [0.8, 5.5] |
| `X_ondev`, ms | **0.05** | [0.00, 0.15] |
| valid spun steps | **≥ 95 %** | |

Reasoning, *estimated* unless marked. The cast is one launch, about 2.2-2.3 µs in situ for the int4 casts (*computed*
from the trims), a little more for 151,936 elements. candle's argmax runs 151,936 iterations on one SM, each with a 64-bit
modulo and division, 15-50 µs; the key kernel is one wave of 16-byte loads plus a launch, 4-6 µs. The bubble the device
loop removes (readback, wake-up, upload, the first ops starving the device) is 20-45 µs. Small pageable host-to-device
copies are usually staged by the driver without waiting, hence the P4 prediction; a wait would show as
`OnDevCandle / F16` falling below `OnDevKey / Key`. E is the gqa-ab decode step (7.44 ms, *computed*). H counts 929
launches, about 859 allocations and frees and 507 kernel lookups, at 2-6 µs an op.

## 7. Decision rules

- **Identity.** Any arm, round or clock pass that returns a different token from `Base` voids the run; the arm is the
  bug, since every gesture claims bit identity. A failed load check, a surviving mutant, or a mutant whose edit did not
  match exactly once voids the run.
- **A1, the cast.** a1 is proposed (`logits_last_native` in the greedy loops, an engine default with no config field, as
  the trims were; the operator's decision) if the range of `F16 / Base` does not lie entirely below 1.000 and P3 reads
  t(`fast_argmax_f16`) ≤ t(cast) + t(`fast_argmax_f32`). Otherwise it is not proposed. The bench cannot resolve a1's
  expected 0.03-0.05 %, so the direct kernel times decide its sign, and the ratio only guards against a surprise.
- **A2, the kernel.** If the range of `Key / F16` lies entirely above 1.000, the device loop takes the key kernel. If it
  lies entirely below 1.000, the device loop keeps candle's argmax. Otherwise P3 decides: the key kernel if
  t(`tok_argmax_h`) < t(`fast_argmax_f16`, direct), candle's argmax if not.
- **A3, the token on the card.** Read on `OnDevKey / Key` if A2 chose the key kernel, on `OnDevCandle / F16` if A2 chose
  candle's. If that range lies entirely above 1.000, the device-token loop with A2's argmax is proposed as the served
  door's default in a follow-up commit, with a served-door measurement under its own prereg; the operator decides. If
  it lies entirely below 1.000, it is not served, and the hidden wait is looked for first in P4. Otherwise it is a curve
  point: nothing served changes, and the operator says whether the code stays as a measurement mode. `OnDevKey / Base`
  moves three mechanisms and decides nothing.
- **B, the host diet, read after a3.** With slack = D − H and `X_ondev` = `E_ondev` − D, medians over valid spun steps:
  - B0: fewer than 90 % of spun steps valid. H and D are not measured, and nothing is proposed on the diet.
  - B1: at least 90 % valid, slack ≥ 0.10 ms and `X_ondev` ≤ 0.10 ms. The host is hidden once a3 lets it run ahead; no
    diet.
  - B2: at least 90 % valid, −0.40 ms < slack < 0.10 ms. The host bounds or nearly bounds the device-token step: b1 (our
    outputs written in place) gets its own prereg and A/B.
  - B3: at least 90 % valid, slack ≤ −0.40 ms. The diet becomes the next gesture, and b2 (a vendored candle with cached
    function handles) goes to the operator.
  - B4: at least 90 % valid, slack ≥ 0.10 ms and `X_ondev` > 0.10 ms. A stall that neither the host span nor the device
    work explains: no diet, and a kernel timeline is proposed to find it.
  - `E_base` − `E_ondev` is reported as the bubble a1 + a2 + a3 recover, and decides nothing.
- **NaN.** The key kernel refuses a generation whose logits hold a NaN, where candle would emit some token. Serving it
  makes that refusal the served behaviour; the operator accepts it or not with A3's proposal.
- Otherwise, the run is void. A kill is the operator's decision.

## 8. Cost

*Estimated* on l40sx1 at $1.80/h. Container and image pull 3 min, oracle 2 min, step 2 about 3 min (dense load 73.7 s,
*measured* in `served-grouped-2026-11-02`), step 3 about 1.5 min. Step 4: the load, about 20 s; P1 and P2, ten NVRTC
compiles and 10 × 41 calls with 12.5 MB uploaded each time, about 10 s; P3 and P4 about 1 s; five arms × seven rounds ×
about 2 s, about 70 s; the E clock, 5 × 2 arms × 65 tokens, about 6 s; the spun loop, 315 steps × (6 × 5 ms + 7.5 ms),
about 12 s. Step 4 in all about 2.5 min. In all about 12 min, **$0.35**. Anchor: `trims-contexts-4b`, 700 s and $0.35 for
a five-arm A/B plus a context sweep (*measured*, `docs/data/jobs.csv`). Timeout 20 min, **$0.60** at worst
(*computed*), the operator's cap; the slowest comparable job, `served-grouped-4b`, billed 18 min. The image build on the Space is free, and its publish needs the operator's go as much as the job does.

Inside the lot job the container start and the image pull are paid once for four parts, so this run's share is its
steps alone, about 9 min with its oracle, **$0.27** (*estimated*). Its in-job ceiling is 18 min, **$0.54** (*computed*),
under the cap of $0.60. The lot job's own timeout and total are in `ops/jobs/lot1.sh`.

## 9. What this run does not establish

- The gestures through the served door, at long context, at 8B or 14B, on another card, or on Metal.
- The side stream for `v_proj` (Part C of the draft): not built in this commit.
- Quality: the gate is token identity, so no score can move.
- Where the device-side fixed cost of the remaining launches goes: no kernel timeline in the image.
- The EOS form of the device-token loop (`bin/gsm8k`, `bin/chat`): not written; it follows an A3 proposal.

## 10. Deviations

Any departure from this file goes in `preregistration-gestures-2026-11-04-ECARTS.md`, beside it, never inside it.

## 11. What changed from the draft of 2026-11-02

- Part C is out: `vstream.rs` was not built, so the arms `VStream`, `BaseEv`, `VStreamEv`, `Bundle`, the race
  pre-check, rule C and their predictions are gone. The cost falls from $0.39 to $0.35.
- The draft said case 1 holds the largest key of the card sequence. It does not: case 2's +inf sits at thread 10,
  bitrev 320, and outranks case 1's at thread 383, bitrev 1018 (*computed*, `the_card_sequence_is_41_cases_with_the_nan_last`).
  An unreset slot (m6) still poisons every case from 3 on.
- A ninth card mutant, m9, swaps the two halves of each 32-bit word of the vector load; the `+inf last` case kills it.
- The NaN case holds a unique maximum at 777 beside its NaN, so a kernel that dropped the flag would return 777, not
  the 0 the draft's uniform row also gave without a flag.
- The E clock is `(T(64) − T(1)) / 63` on whole passes, where the draft synchronized after the prefill and timed from
  the first decode enqueue. It needs no hook in the loop, and both ends of each pass end on a readback.
- Events go off before the load, as the draft said and as the served door does, and not after it as `LLVQ_GQA_AB`
  does: cudarc records a read event at every use of a slice created while tracking was on, whatever the flag says at
  the use (`SyncOnDrop::record_event`, cudarc 0.19.8 `core.rs:1121`), so a mode that turned it off after the load would
  run its weights in a state the served door never runs in.
- `LLVQ_KERNEL_DIR` is refused beside the mode, so the text the load check certifies is the embedded one.
- Gate 5 keeps the draft's fix: it compares the served door's extracted text with the digest taken by the same door.
- Two host tests the draft did not have: the recording key, added because the tiny model's greedy text settles into a
  two-token cycle and let two loop mutants survive, and the a1 test on the f16 logits against their cast.
