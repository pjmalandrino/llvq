DRAFT, not stamped, not committed. Wave-5 deep dive "gestures", 2026-11-02, revised after review. It assumes code that does not exist yet (`gestures.md` §6, steps 2 to 4); every "to fill" below is filled from the commit that carries it, before the stamp. Proposed path: `proofs/preregistration-gestures-YYYY-MM-DD.md`.

# Preregistration. Cheap bit-exact gestures at the token boundary, one mechanism an arm, with direct probes and the host clocks; the side stream on the fused graph only

**Written on YYYY-MM-DD, to be committed and TIMESTAMPED BEFORE the run.** Operator go to launch: none yet. Cost: about **$0.35** on l40sx1 with Part A alone, **$0.39** with Part C, *estimated* below (§8); timeout 45 min, so $1.35 at worst. Cap for this wave: none in force (ETAT §5 says one is owed before the next paid job); the operator sets it. Spent before it: $248.84 over 237 rows of the registry (*measured*, `docs/data/jobs.csv`), plus whatever the runs of `prereg-gestures-0.md` and `-2.md` spend first.

**Base commit:** `<sha to fill>`. **Part C runs** only if that commit carries the operator's norm+rotation fusion (step 2 of `docs/archive/audit-debit-2026-10-29.md` §5), named here by its commit `<sha to fill>` and its design, (i) or (ii) of §1. If it does not, Part C's arms are not built into the mode, the side stream is never created, and this run measures Part A alone.

## 1. Why this run exists

The 4B decodes at 133.3 tok/s through the served door under the grouped config, 7.50 ms a token (*measured*, `docs/mesures/served-grouped-2026-11-02.txt`). Three cheap levers have never been measured.

- **The token boundary.** Each token pays an f16-to-f32 cast of the logits, candle's one-CTA `fast_argmax` with a pageable upload of its dims and strides on every call (`cuda_backend/mod.rs:333`), a blocking readback and an upload of the id (`model.rs:2141-2147`). Fenced, that boundary costs 0.078-0.101 ms (*measured*, `phases-2026-08-07.txt`). The audit of 2026-10-29 listed the logits cast and the id kept on the card in its step 0 bis; the trims of 2026-10-30 did its other items.
- **The int4 `v_proj` waits on one stream.** In today's graph it reads the unrotated ln1 output, yet runs after the one-CTA rotation (8.05 µs at n = 2,560, *measured*, `rotation-cuda-2026-08-05.txt`), q and k (`rotplan.rs:519-545`). The operator's fusion branch moves that window. Under design (i), the fused norm+rotation kernel also writes v's input; sm_89 has no programmatic dependent launch, so the fork moves after it and v overlaps q and k only. Under design (ii), v's int4 kernel takes ln1 in its prologue, and the fork stays before the one-block kernel. That is why (c) is measured only on the fused graph.
- **The host's share.** No journal holds the host span of the served step. Two facts suggest it is hidden, a reading that is *estimated* and rests on two assumptions. First, the graph replay of 2026-09-01 saved 1.192, 1.204 and 1.242 ms a token at 4B, 8B and 14B while the eager step went from 10.08 to 21.32 ms (*computed* from the three A2 journals); that step had about 1,040 launches (≈750 candle, 288 ours, `a2-verdict-2026-09-01.txt`), a Prealloc(256) store, a fresh stream and events off, and today's step is 7.44 ms with 929 launches, about 0.74 of it, so the margin it rests on is smaller now. Second, the events trim removed thousands of driver calls a token for 0.0945 ms (*computed*, `trims-contexts-2026-10-30.txt`), which implies a hidden host only if each removed call cost well over 0.1 µs of host time, which no journal measures; `cuStreamWaitEvent` also puts commands on the device. The clocks of §3 are the only decider.

## 2. What the code now holds

The commit that carries this file adds, all off by default, so every published path runs as before (`code/gestures/` holds the sketches):

- `Qwen3::generate_loop` with four kinds besides `generate` itself, each one mechanism from its neighbour (`decode_on_device.rs`): `F16` (no cast), `Key` (the key kernel into a one-slot ring, `to_scalar` of its rank-0 view, `from_slice`), `OnDevKey` (no per-token readback or upload; one readback at the end), `OnDevCandle` (candle's argmax output embedded as the next input, no copy).
- `tok_argmax_h` (a stand-alone kernel source, `tok_argmax.cu`): a 64-bit key that reproduces candle 0.9.2's tie order, 75 CTAs of 256 threads, written into the ring through a candle `InplaceOp2`; a NaN raises a sticky flag and the loop refuses the generation. Its load check runs 41 calls in one sequence against candle's own `fast_argmax_f16` on the card (8 adversarial cases, 32 sparse tie sets, one NaN).
- `LLVQ_GESTURE_AB=1` in `bin/fusedrun` (`gesture_ab.rs`), refused beside `LLVQ_CONFIG`, beside any other mode, and with `LLVQ_KV_PREALLOC`. It runs, in order: the load check; the eight kernel mutants m1-m8, each applied in memory to the embedded text with every `from` counted exactly once, compiled as its own NVRTC module and run through the same sequence; the kernel-time probe; the pageable-copy probe; the arms; the clocks.
- Part C only: one non-blocking side stream, `Proj::forward_forked` and the join (`vstream.rs`), the arms `VStream`, `BaseEv`, `VStreamEv`, `Bundle`, the race pre-check with its no-join arm.

*Measured* on the Mac CPU (to fill): every kind returns `generate`'s tokens on `Cat`, `Grow` and `Prealloc`, under the three forms, in f32 and f16, with groups of 2, 4 and 5; the key returns the replayed candle index on every case of `argmax_key_ref` (300 sparse cases in the release loop); N host mutants, each killed by a test.

## 3. What runs

One image built from the base commit. One job, l40sx1, the sealed 4B (`sealed-4b-2026-09-23/qwen3-4b-sealed.bin`, sha256 `886391a8…`), configs `qwen3-4b-tetra-e4.json` (`098fe49c…`) and `qwen3-4b-tetra-e4-grouped.json` (`2455fe85…`), written by the job from its launch command and checked by sha256.

1. **Oracle** on Qwen3-0.6B, rule 10.
2. **The bench door, replayed.** `fusedrun <file> 256` with the served flags spelled out and `LLVQ_EMBED=q4`, against the dense arm, as in run 1 of the paper table. It checks that the default path did not move.
3. **The served door, grouped, unchanged.** `LLVQ_CONFIG=<e4-grouped> fusedrun <file> 256`. It checks that the served path did not move: the gestures are not served by this commit. Its text is extracted as `ops/jobs/served-grouped.sh:133-134` does, from the `served : ` line to `No dense arm on this path`.
4. **The A/B.** The served flags plus `LLVQ_GESTURE_AB=1`, 256 new tokens after the 5-token prompt, the model in the served 4B state (`Grow(256)` + grouped), events off before any allocation.
   - P1, the load check; P2, the kernel mutants.
   - P3, the kernel-time probe, on the f16 logits of the prompt's last position, 5 × 100 back-to-back launches each, event-timed: candle's `fast_argmax_f16` and `fast_argmax_f32` launched directly from candle's module with their info buffer uploaded once; the logits cast through candle; `tok_argmax_h`; and candle's full `argmax` call.
   - P4, the pageable-copy probe, five times each behind `spin_ns(5 ms)` on candle's stream: the host time of `Tensor::from_slice` of one id, of candle's `argmax` on the logits vector, and of `to_scalar` on one u32 (the control, which must wait).
   - Part C only, the race pre-check: 16 tokens of `VStream` with `spin_ns(50 µs)` on the side stream before each v, which must equal `Base`; then 16 tokens of the no-join mutant with the same delay, which must diverge, then a context synchronize.
   - The arms, every arm in every round, 2 rounds discarded and 5 timed: Part A `Base`, `F16`, `Key`, `OnDevKey`, `OnDevCandle`; Part C `VStream`, `BaseEv`, `VStreamEv`, `Bundle`. `BaseEv` and `VStreamEv` re-enable cudarc's tracking before their allocations and disable it after, as `LLVQ_TRIM_AB` does (`fusedrun.rs:905-925`).
   - The clocks: `E_base` and `E_ondev` from unspun passes (5 × 64 tokens each, decode steps only); then the spun `OnDevKey` loop, 5 × 64 steps, six segments a step each behind `spin_ns(5 ms)` (the step start, then layers 6, 12, 18, 24, 30 through the capture hook), giving H, the host enqueue span, and D, the device time, valid only if every segment was enqueued at least 1 ms before its spin could end.
   - Part C only: one `VStream` pass of 64 tokens timing v by events on the side stream.

Every arm includes the 5-token prefill in its clock, as the trims A/B did: about 23 ms of about 1.9 s (*computed* from `served-grouped` and `gqa-ab`), so every ratio is diluted toward 1 by about 1 %, and that is declared here.

## 4. Gates, each one voids the run (Part C's gates void Part C only)

1. The image builds, and `oracle` prints MATCH.
2. The sealed file and both configs at their bytes and sha256 on the mount.
3. Every load prints `int4: tv_q4_hh equals tv_q4_h bit for bit`; the header of every step reads `tile 64`; no `LLVQ_*` variable is in the job environment.
4. Step 2 prints `fused F0: 256 tokens identical to the dense arm`, exactly one `  fused F0:  ` line, and that line hashes to `8784ffc423bfbeeaeb9a92abd426573cc45f9a083ba56e61e52d9ff40e3b018f`.
5. Step 3 prints `attention grouped on grow(256)` on its `served config: ` line, `No dense arm on this path`, exactly one `served : ` line, and its extracted text hashes to `0597cee8b20befecdf92c2f3f2f1938d8216bd244542faab42d2637920a3c12a`, the text both configs served on 2026-11-02 (*measured*, `docs/mesures/served-grouped-2026-11-02-brut/lines.sha256`). The first draft compared this text with step 2's line, which the two doors print in different formats.
6. Step 4 prints the load-check line (`40 cases in sequence, NaN flag raised`), the mutant line (`8 of 8 killed on the card`, each `from` counted once), the event-tracking line (`off (one stream, ordered by issue)` for Part A alone, `off (two streams, ordered by explicit events)` with Part C), the probe and clock lines, and its arm table in full. The mode refuses to print the table if any arm in any round returns a token that differs from `Base`.
7. Part C: the race pre-check line reads the joined arm identical to `Base` and the no-join arm diverged within 16 tokens.

## 5. Outcomes reported, never gated

- Each arm's tok/s, median of 5 rounds with its range; the seven single-mechanism ratios round by round: `F16/Base` (a1), `Key/F16` (a2), `OnDevKey/Key` and `OnDevCandle/F16` (a3), and with Part C `VStream/Base`, `BaseEv/Base`, `VStreamEv/BaseEv`; `OnDevKey/Base` and `VStreamEv/Base`, labelled as several mechanisms.
- P3: µs a launch for each of the five; P4: the three host times in ms, with the control.
- E_base, E_ondev, H and D in ms a step, slack = D − H, X_ondev = E_ondev − D, E_base − E_ondev, and the count of valid spun steps.
- Part C: v's in-situ time, median over 36 layers and 64 tokens.
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
| P4 `from_slice` behind the spin, ms | **< 0.5** (fire-and-forget) | |
| P4 candle `argmax` behind the spin, ms | **< 0.5** (fire-and-forget) | |
| P4 `to_scalar` control, ms | **5.0** | [4.5, 5.6] |
| E_base, ms a decode step | **7.44** | [7.20, 7.70] |
| E_ondev, ms a decode step | **7.37** | [7.10, 7.65] |
| H, ms a step | **3.8** | [2.0, 6.5] |
| D, ms a step | **7.30** | [7.00, 7.60] |
| slack D − H, ms | **3.5** | [0.8, 5.5] |
| X_ondev, ms | **0.05** | [0.00, 0.15] |
| valid spun steps | **≥ 95 %** | |
| Part C, design (i): `VStream / Base` | **1.008** | [0.990, 1.025] |
| Part C, design (ii): `VStream / Base` | **1.025** | [1.005, 1.045] |
| Part C: `BaseEv / Base` | **0.975** | [0.960, 0.990] |
| Part C: v in situ, µs a layer | **7** | [4, 12] |

The design row that does not match the base commit is struck before the stamp.

Reasoning, *estimated* unless marked. The cast is one launch, about 2.2-2.3 µs in situ for the int4 casts (*computed* from the trims), a little more for 151,936 elements. Candle's argmax runs 151,936 iterations on one SM, each with a 64-bit modulo and division on `size_t` dims (`cuda_utils.cuh:25-38`), 15-50 µs; the key kernel is one wave of 16-byte loads plus a launch, 4-6 µs. The bubble the device loop removes (readback, wake-up, upload, the first ops starving the device) is 20-45 µs. Small pageable host-to-device copies are usually staged by the driver without waiting, hence the P4 prediction; a sync would show in `OnDevCandle / F16` falling below `OnDevKey / Key`. E is the gqa-ab decode step (7.44 ms, *computed*). H counts 929 launches, about 859 allocations and frees and 507 kernel lookups, at 2-6 µs an op. Part C: in today's graph v would overlap the 8.05 µs rotation, 36 × (5-9 µs − a join of 1-2 µs); under design (i) it overlaps q and k only. `BaseEv / Base`: the one-stream allocation events cost 1.15 % (*measured*, `trims-contexts`), and multi-stream tracking adds a wait and a record per slice argument, 3.67 against 3.63 µs a launch on a near-empty chain (*measured*, `a3-graph-2026-08-06.txt`), so 1.7-3.2 % in all (*computed* from those inputs with one to four slice arguments a launch, an *estimated* count).

## 7. Decision rules

- **Identity.** Any arm or round that returns a different token from `Base` voids the run; the arm is the bug, since every gesture claims bit identity. A failed load check, a surviving mutant, or a mutant whose `from` did not count exactly once voids the run.
- **A1, the cast.** a1 is proposed (`logits_last_native` in the greedy loops, an engine default with no config field, as the trims were; the operator's decision) if the range of `F16 / Base` does not lie entirely below 1.000 and P3 reads t(`fast_argmax_f16`) ≤ t(cast) + t(`fast_argmax_f32`). Otherwise it is not proposed. The bench cannot resolve a1's expected 0.03-0.05 %, so the direct kernel times decide its sign, and the ratio only guards against a surprise.
- **A2, the kernel.** If the range of `Key / F16` lies entirely above 1.000, the device loop takes the key kernel. If it lies entirely below 1.000, the device loop keeps candle's argmax. Otherwise P3 decides: the key kernel if t(`tok_argmax_h`) < t(`fast_argmax_f16`, direct), candle's argmax if not.
- **A3, the token on the card.** Read on `OnDevKey / Key` if A2 chose the key kernel, on `OnDevCandle / F16` if A2 chose candle's. If that range lies entirely above 1.000, the device-token loop with A2's argmax is proposed as the served door's default in a follow-up commit, with a served-door measurement under its own prereg; the operator decides. If it lies entirely below 1.000, it is not served, and the hidden synchronization is looked for first in P4. Otherwise it is a curve point: nothing served changes, and the operator says whether the code stays as a measurement mode. `OnDevKey / Base` moves three mechanisms and decides nothing.
- **B, the host diet, read after a3.** With slack = D − H and X_ondev = E_ondev − D, medians over valid spun steps:
  - B0: fewer than 90 % of spun steps valid. H and D are not measured, and nothing is proposed on the diet.
  - B1: at least 90 % valid, slack ≥ 0.10 ms and X_ondev ≤ 0.10 ms. The host is hidden once a3 lets it run ahead; no diet.
  - B2: at least 90 % valid, −0.40 ms < slack < 0.10 ms. The host bounds or nearly bounds the ondev step: b1 (our outputs written in place) gets its own prereg and A/B.
  - B3: at least 90 % valid, slack ≤ −0.40 ms. The diet becomes the next gesture, and b2 (a vendored candle with cached function handles) goes to the operator.
  - B4: at least 90 % valid, slack ≥ 0.10 ms and X_ondev > 0.10 ms. A stall that neither the host span nor the device work explains: no diet, and a kernel timeline is proposed to find it.
  - E_base − E_ondev is reported as the bubble a1 + a2 + a3 recover, and decides nothing: a diet decided on base's exposure could prescribe work a3 already made useless.
- **C, the side stream, Part C only.** If the operator granted events off with two streams before the run, C reads `VStream / Base`. If the operator refused it, the fallback is what the served door would become, so C reads `VStreamEv / Base`, formed round by round, with its two single-mechanism parts reported (`BaseEv / Base`, the price; `VStreamEv / BaseEv`, the fork under tracking); the ruling, not the A/B, binds those two mechanisms. If no ruling exists at the run, both are reported and C applies when the ruling comes. On the ratio so chosen: a range entirely above 1.005 proposes forking `v_proj` for the served door; a range entirely below 1.000 means it is not served; any other range is a curve point. A failed race pre-check voids Part C.
- **The bundle** is reported. Serving decisions are per gesture, and a gesture is served only with the bundle's identity intact.
- Otherwise, the run is void. A kill is the operator's decision.

## 8. Cost

*Estimated* on l40sx1 at $1.80/h. Container and image pull 3 min, oracle 2 min, step 2 about 3 min (dense load 73.7 s, *measured* in `served-grouped-2026-11-02`), step 3 about 1.5 min. Step 4, Part A: load 10 s; P1 and P2, nine NVRTC compiles and 9 × 41 calls, about 5 s; P3 and P4 about 1 s; five arms × seven rounds × about 1.9 s, about 67 s; the unspun clocks about 6 s; the spun loop 5 × 64 × (6 × 5 ms + 7.5 ms), about 12 s. With loading, about 2 min. In all about 11.5 min, **$0.35**. Part C adds four arms × seven rounds, the pre-check and the v pass, about 1 min, **$0.39**. Anchor: `trims-contexts-4b`, 700 s and $0.35 for a five-arm A/B plus a context sweep (*measured*, `docs/data/jobs.csv`). Timeout 45 min, **$1.35** at worst.

## 9. What this run does not establish

- The gestures through the served door, at long context, at 8B or 14B, on another card, or on Metal.
- That events off with two streams is safe on every path of the engine: Part C's arms are safe by their explicit events; the served door needs a review.
- Quality: the gate is token identity, so no score can move.
- Where the device-side fixed cost of the remaining launches goes: no kernel timeline in the image.
- (c) in any graph but the base commit's: a later change of the fusion design voids its numbers.

## 10. Deviations

Any departure from this file goes in `<this file>-ECARTS.md`, beside it, never inside it.
