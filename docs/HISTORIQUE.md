# History

The project's chronological thread, one entry per period, from 2026-07-24 to 2026-11-12. The current state is in [ETAT.md](ETAT.md), the lab rules in [METHODE.md](METHODE.md), the writing rules in [STYLE.md](STYLE.md). Entries are appended at the bottom and the past is not rewritten.

## 2026-07-24 to 07-28. Foundations, G1 to G4

- Golay [24,12,8] and Λ₂₄ held by exact invariants: kissing number 196,560, theta series, N(13) = 280,974,212,784,720 (*computed*, `classes_reproduce_theta_series`).
- Exact NN search for m ≤ 13 and bijective 48-bit indexing, format v1 (Golay generator `0xC75`, codeword order, class order).
- G4: 92.23% retention at 1.9999 b/dim, β* = 0.350, MSE 0.0775 (*measured*, `llvq-bench`, 20,000 blocks). The paper gives 89.37% with a β detuned by ±0.04.
- Encoder: 639 µs/block/core, 5.5× the start; `nearest_angular` 680 µs (*measured*, `encbench`). Telescoping sums by runs: one class in ≤ 5 operations.
- In-house forward pass against candle: max |Δhidden| = 0 (*measured*, `bin/oracle`). Qwen3-0.6B FP32: ppl 19.1481 over 73 windows (*measured*).
- Smoke 0.6B/28 blocks: ×1.811 with rotation against ×2.290 without, ×2.748 with `group_scales` (*measured*, `bin/smoke`, 131k tokens).
- Decisions: shape–gain rather than spherical shaping; `TailPolicy::KeepExact`; 4 Hessians per block; `group_scales` disabled; single-variable A/B on 3 blocks.
- The multi-type sweep of the parity repair is dead code: the maximum is always at j = w−1 (found by mutation).
- Text extraction from the paper's PDF is corrupted; the image rendering is reliable, transcribed in [llvq-paper-notes.md](llvq-paper-notes.md).

## 2026-07-29 to 07-31. G5, first 4B

- On 07-31, the first 4B announced at 2.0653 then 2.1117 b/weight (cap 13, 14.9104 ppl) is refuted. The free 16-bit magnitude was not being charged: real values 2.7289 and 2.7338 (*computed*, [archive/retraction-et-gain.md](archive/retraction-et-gain.md)).
- Sealed file `leech1c12` (cap 12, 47 + 1 bits): 16.9617 ppl at 2.1696 b/weight (*measured*, `~/llvq-run-4b-artefact.log`). It weighs 981 MB of projections, 1.771 GB with the f16 embedding. The 2.0702 effective b/weight is the ideal rate printed by smoke; it does not describe the written file (see [fiche-4b.md](fiche-4b.md)).
- G5 gate: QTIP 17.04 at 2.000 (paper). Green with 8.5% more bits.
- On 07-31, "quantizing the gain costs almost nothing" is refuted: the A/B was comparing an arm with itself (difference 7.1e-15). Correct value: +3.17% of ppl for −0.618 b/weight (*measured*, 0.6B, 3 blocks).
- 4B FP32 baseline 12.2336 against 12.41 in the paper, 12 windows against 73 (*measured*).
- Calibrating on C4 gives 14.91 against 15.29 on wikitext (*measured*, [fiche-4b.md](fiche-4b.md)): "in-domain calibration flatters by 12%" is refuted, the gap was measuring the corpus difficulty.
- 4B in 3.45 h with `faer`, 6.3 h without (*measured*).

## 2026-08-01. Audit A

- Zero dtype confounder: ppl f32 against f16 within 0.1% (*measured*). Sealed file decoded in f16: 16.9415, ×1.3846 (*measured*, `bin/ppl`), fingerprint `3f1baca9033bf251`.
- MMLU reported in micro, as the paper does. The prediction "drop of ~−10 pp after correction" is refuted on 08-02: the aggregation was worth 0.93 pp (*computed*, [mmlu-micro-2026-08-02.log](archive/mmlu-micro-2026-08-02.log)).
- G4 benchmark re-anchored on `LeechShapeGain` (gain coded on the norm): shape–gain 0 bit 88.90%, MSE 0.0850 (*measured*, `llvq-bench`).
- `bin/thesis` on Metal: FP16 21.69 ms against LLVQ 10.46 ms, 252 projections, 1,105,920 rows checked against f64 (*measured*, [mesures/thesis-temoin-2026-08-04.txt](mesures/thesis-temoin-2026-08-04.txt)).
- The published 2.07× is the top of a range [2.029; 2.080] over three invocations (*measured*, [mesures/thesis-temoin-2026-08-04.txt](mesures/thesis-temoin-2026-08-04.txt)).

## 2026-08-02 to 08-04. MMLU micro, 8B, 32B, single shell

- MMLU micro 4B on Metal: 70.42 ± 1.28 against 56.09 ± 1.36, −14.33 pp (*measured*, [mmlu-micro-2026-08-02.log](archive/mmlu-micro-2026-08-02.log)); baseline +0.22 pp from the paper. Replaced by the L40S measurement of 08-06.
- Per-subject profile: abstract algebra and accounting at chance (25%), history and law above 80%. 2-bit damages reasoning more than recall.
- 8B `leech1c12L3` on HF Jobs: ×1.267 at 2.0436 b/weight, 4.18 h, $11.48 (*measured*). Superseded as a scale point by the requantification of 08-08.
- 32B de-risked on 4 blocks: 621 s/block against ~500 predicted, $5.43 (*measured*); full run re-quoted at ~11.4 h and ~$62 (*estimated*). C3 (bf16) is a prerequisite.
- On 08-04, "the single shell beats the union" is refuted: at 49 bits, union 0.0725 against shell 13 at 0.0762 (*measured*, `llvq-bench`). The 92.24% retention was dividing by a fractional bit rate that no file pays.
- Λ₂₄(12) has 301 classes, not 383 (*computed*, `enumerate_classes`).
- On 08-03, the "690 packages" is corrected: 261 for `llvq-llm`, 3 for `llvq-artifact` (*measured*).

## 2026-08-05. Batch K-1, CUDA port

- Metal ladder under a single accounting, 7 rounds of which 2 discarded: Slot32 5.510 b/weight 2.03× [2.03–2.10], Flat32 5.256 0.91×, Grouped32 3.498 0.69× (*measured*, [mesures/k1-metal-2026-08-05.txt](mesures/k1-metal-2026-08-05.txt)).
- The old ladder "3.35 nested 0.68×; 4.54 Flat32 0.90×; 5.51 Slot32 2.07×" mixed several accountings. Superseded.
- Same binary, three invocations: 2.029×, 2.050×, 2.080×. Rule: publish a range, and form the ratio round by round.
- The predicted benchmark conflict does not exist on Apple; `float4` gives 3.5% and 5.1% on both sides (*computed*, [mesures/k1-metal-2026-08-05.txt](mesures/k1-metal-2026-08-05.txt)).
- L ≤ 4 cap: ≤ 4.7083 b/weight, 4,708,799 groups out of 4,708,800 carry a 4-level block (*computed*).
- CUDA rotation kernel written, 15 mutants killed, never run on a card that day.
- Attribution of the CUDA headroom: 2.04 ms/token, stream 33%, latency and decoding 59% (*measured*, [mesures/attribution-cuda-2026-08-05.txt](mesures/attribution-cuda-2026-08-05.txt)). The split of that 59% into latency-occupancy 39% (0.803 ms) and residual decoding 19% (~0.396 ms) is measured the same day by varying the occupancy (*measured*, [mesures/fusion-qkv-cuda-2026-08-05.txt](mesures/fusion-qkv-cuda-2026-08-05.txt)). Requalified on 08-21 as a property of our geometry.
- Decision: Metal first (free benchmark), CUDA next (reproducible); `wgpu` never.

## 2026-08-06. Batch A, the kernel in the model

- `fusedrun` Slot32 on L40S: 47.0 tok/s in 3.28 GB against 43.5 in 8.04 dense, 88 identical tokens (*measured*, [archive/passation-lot-a-2026-08-06.md](archive/passation-lot-a-2026-08-06.md)). Single point, superseded by B2.
- 4-arm 4B campaign: MMLU f16 70.32 ± 1.28, AWQ 70.04, LLVQ 55.59 ± 1.35; ppl ×1.105 against ×1.384 (*measured*, [mesures/a4-campagne-2026-08-06.txt](mesures/a4-campagne-2026-08-06.txt)). On a 4B, 4-bit dominates everywhere except on disk.
- The quantized arm loses 0.50 pp between Metal (56.09) and CUDA (55.59): provenance debt, the log predates the fingerprints.
- C1: Planes14 1.14× [1.14–1.15] faster than Slot32 at 4.804 b/weight, identical content (*measured*, [mesures/c1-planesbench-2026-08-06.txt](mesures/c1-planesbench-2026-08-06.txt)). Served the same day: 48.7 tok/s in 2.96 GB (*measured*, [mesures/planes14-fusedrun-2026-08-06.txt](mesures/planes14-fusedrun-2026-08-06.txt)).
- Batch B, 0.6B 3 blocks (*measured*, [archive/verdicts-lot-b-2026-08-06.md](archive/verdicts-lot-b-2026-08-06.md)): cross-seed σ 0.7%; oracle −1.6%; volume −1.2% for ×13. Damping 0.35%; L ≤ 4 swap +4.75%.
- The ×100 calibration run is buried; L ≤ 4 is dead on quality. The 0.7% σ will be refuted on 08-19 at the published size.
- Batch A errata: "5.51 against 4.50" is banned, two denominators and two four-bits. Rule: b/param over the whole model, embedding included.
- On 08-06, "5 gates out of 7", "the kernel is not wired in" and "the next decision point is C1" are superseded.

## 2026-08-07. Design C, Golay70, q8 embedding

- Design C: ×1.99 of ppl at 28 blocks (35.98 → 71.42), automatic gate, $0 (*measured*, [archive/verdicts-nuit-2026-08-07.md](archive/verdicts-nuit-2026-08-07.md)). Refuted; norm rigidity is load-bearing at depth.
- Golay70 v1: 3.589 b/weight, 1.31× [1.29–1.32], 195 GB/s against a criterion of 1.6× (*measured*, [mesures/e2-golay70-bench-2026-08-07.txt](mesures/e2-golay70-bench-2026-08-07.txt)). Dropped.
- Same benchmark: Slot32 1.87× [1.86–1.88] 428 GB/s; Planes14 2.14× [2.11–2.15] 425; Planes12x 4.342 b/weight 1.98× [1.95–1.99], exact quality.
- q8 embedding in production: ppl 16.9358, MMLU 55.70 (*measured*, [mesures/campagne-finale-bras4-2026-08-07.txt](mesures/campagne-finale-bras4-2026-08-07.txt)). The journal gives `fusedrun` at 88.4 tok/s in 2.60 GB as displayed; the campaign summary writes 88.4-88.5 (*measured*, [campagne-finale-2026-08-07.md](archive/campagne-finale-2026-08-07.md)), a single point.
- Mechanism of the throughput jump: our dense arm copies 778 MB of vocabulary per token, ~26 ms (*measured*, [mesures/phases-2026-08-07.txt](mesures/phases-2026-08-07.txt)), `Head::project` → `broadcast_matmul`.
- Rule: always two throughput formulations, raw and same-head. The speed-against-size dilemma is lifted: Planes14 is smaller and faster than Slot32.
- "The format ladder is closed" is written that day; it reopens on 08-10.

## 2026-08-08. 4B→8B scale, one variable

- 8B `leech1c12`, same codebook and same corpus: ppl ×1.2201, MMLU 76.08 ± 1.21 → 65.52 ± 1.31, −10.56 pp (*measured*, [mesures/campagne-8b-qualite-2026-08-08.txt](mesures/campagne-8b-qualite-2026-08-08.txt)). Gap to 4-bit 14.45 → 7.49 pp.
- 8B speed: dense 26.5, f16 34.4 (×1.30), q8 69.3 tok/s (×2.61) in 5.45 GB (*measured*, [mesures/campagne-8b-q8-2026-08-08.txt](mesures/campagne-8b-q8-2026-08-08.txt)). Single points, replaced by B2.
- Untied heads: 2.49 GB of tables in f16; sealed file 4.32 GB f16, 3.157 GB q8 (*measured*). Without q8, the 8B reverses nothing.
- `codebook_fingerprint` pinned at `0x338f_420f_1186_6319`; `forbid(unsafe_code)` set on `llvq-artifact`; unconditional `#[ignore]` for the archives (11 min 26 s → 2.3 s).
- "Full suite ~45 s" is refuted: tens of minutes (*measured*, 17 min without finishing the first crate). "Seven crates" becomes eight, "unsafe exclusive to llvq-llm" becomes metal 12, cuda 13, llm 11.
- "26 min of download over 65.5 GB" is a circular number; the out-of-loop part is bounded at ≤ 846 s (*computed*).

## 2026-08-09. Planes12x wired, 5.162 b/param

- `rtbits`: 4B Planes14 + q8 = 5.162 b/param, below AWQ at 5.302; 8B 5.322 against 5.956 (*computed on measured bytes*, [mesures/rtbits-planes-8b-2026-08-09.txt](mesures/rtbits-planes-8b-2026-08-09.txt)).
- The "5.11" (embedding at 8 bare bits) and the "≈ 5.15" (card display 2.60 GB) are superseded; a q8 g64 embedding costs 8.5 b/param.
- Planes12x wired into `LLVQ_FUSED_LAYOUT`: 5,096,688 exceptions (3.3824%) over 150,681,600 blocks (*measured*); transcoding 404 s against 84 s, ×4.8 (*measured*, M3 Max 16 threads). Planes12x + q8: 4.745 b/param (*computed*, [mesures/rtbits-planes-8b-2026-08-09.txt](mesures/rtbits-planes-8b-2026-08-09.txt)).
- Not default at 8B: the VRAM is already won there (~11% below AWQ), the throughput would cost ~7% (*estimated*).
- "A candle path" is refuted: the path is ours, sent upstream (candle#3871).

## 2026-08-10. AWQ on the benchmark, the 14B

- AWQ ported into our benchmark: 584 GB/s, 3.38×, 88% of its byte bound against 65% for us (*measured*, [mesures/six-arm-awq-2026-08-10.txt](mesures/six-arm-awq-2026-08-10.txt)).
- E2's 1.6× speed criterion is superseded; E2 reopened on the memory axis with a 2.0× threshold timestamped the next day ([../proofs/preregistration-2026-08-11.md](../proofs/preregistration-2026-08-11.md)).
- 14B: ppl ×1.1894, MMLU 78.97 ± 1.19 → 72.12 ± 1.24, −6.85 pp, paired CI95 [+4.52; +9.12], McNemar 8.7e-16 (*measured*, [mesures/campagne-14b-qualite-2026-08-10.txt](mesures/campagne-14b-qualite-2026-08-10.txt)). AWQ 78.21.
- The AWQ−LLVQ gap of 6.09 pp is written as a bare difference; paired on 08-17.
- "The curve has a knee" and "−43% then −14%" are written on bare points; requalified on 08-17 by metric.
- Three points do not make a scaling law; the 32B would settle it.

## 2026-08-11. Golay70 v2

- v2 decoder (coset logic hoisted to the block): 1.77× [1.76–1.78], 263 GB/s, 1.32× over v1, 40% of the byte bound (*measured*, [mesures/golay70-v2-sept-bras-2026-08-11.txt](mesures/golay70-v2-sept-bras-2026-08-11.txt)).
- Not adopted: below the 2.0× threshold ([../proofs/preregistration-2026-08-11.md](../proofs/preregistration-2026-08-11.md)). No lead left with the format unchanged.
- Chain: prereg 09:30:36, `.ots` 09:31:06, measurement 13:34:31 (*measured*, git).
- `golay70` wired into `LLVQ_FUSED_LAYOUT`: measurable, not served.

## 2026-08-12. Paper, external audit, overhaul

- Paper trimmed by 16% of its words, tag `paper-v1` (*measured*).
- External audit: ~40 numbers retraced, the $22.83 cost recomputed exactly. The 14B point is missing from the paper, from the README and from `CLAUDE.md`. "25% less memory at 8B" is refuted, actual ~11% (*computed*).
- Verdict: the kernel axis stops, the asset is the paper and the quality. Reopened later by the P1→P7 plan, then by phase A.
- Documentation overhaul: 36 documents moved to `docs/archive/`, `HISTORIQUE.md` created as the single thread, `PLAN.md` as the follow-up.

## 2026-08-12 (continued). Batch X, E1c and E3

- E1c14 and E1c12, transposed onto the group of 32 blocks: full sweep of 150,681,600 blocks exact, 401 s (*measured*, [mesures/e1c-sweep-4b-2026-08-12.txt](mesures/e1c-sweep-4b-2026-08-12.txt)).
- Unaligned bits: 4.5551 and 3.7618 b/weight in kernel accounting (*measured*, [mesures/rtbits-e1c-4b-2026-08-12.txt](mesures/rtbits-e1c-4b-2026-08-12.txt)). Superseded on 08-15: the served matvec does not read that accounting.
- X3 thresholds set: ≥ 2.05× replaces Planes14, ≥ 1.9× replaces Planes12x, < 1.6× closes. Set in unaligned accounting, to be re-anchored.
- E3 buried on paper: best point 3.0444 b/weight against a criterion of 2.60 (*measured*, [mesures/radixstudy-x4-2026-08-12.txt](mesures/radixstudy-x4-2026-08-12.txt)). The point inside its class costs 41.50 of the 47 bits.
- MoE: 31.4% of the (layer, expert) cells of gpt-oss-20b are below full rank at 131k tokens (*measured*, [mesures/moe-routing-gptoss20b-2026-08-12.txt](mesures/moe-routing-gptoss20b-2026-08-12.txt)); covering 90% requires ×12.

## 2026-08-13. Paired replay, 4B and 8B

- Six arms replayed to the hundredth, fingerprint `65dcd53655e8bfa5`, $1.30 (*measured*, [mesures/mmlupair-4b-8b-2026-08-13.txt](mesures/mmlupair-4b-8b-2026-08-13.txt)).
- AWQ − LLVQ: +14.45 [+11.60; +17.27] at 4B, +7.49 [+5.28; +9.70] at 8B, disjoint CIs. f16 − LLVQ: +14.73 and +10.57, CIs overlap.
- f16 − AWQ at 4B: +0.27 [−1.63; +2.13], unresolved in micro; +1.97 [+0.92; +3.02] unweighted. The paper's sentence holds in one accounting only.
- Paired SE between different models: 0.79 to 1.44 pp (*measured*).

## 2026-08-14 to 08-15. int8 KV cache

- KV q8, $0 and ~2 h 45 min of Mac time: ppl +0.049% [−0.071; +0.170], MMLU +0.33 pp [−0.45; +1.22], McNemar p = 1.0000 (*measured*, [mesures/kvq8-4b-2026-08-15.txt](mesures/kvq8-4b-2026-08-15.txt)).
- Throughput 0.927× and 0.945× at n_new = 128, the 1024 series abandoned (661 s against 600). Shipped, not default: short context only. KV memory ÷1.882 (*computed*).
- f16 control: 16.9415 and 56.09%, reproduced to the ten-thousandth and identically.
- Bar of a constant-file A/B: ±0.12% in ppl, SE 0.43 pp in MMLU (*measured*). The "McNemar σ 0.4-0.6 pp", never computed, is superseded.
- Preregs P2 to P5 rewritten after an adversarial review with 18 blockers. MoE (P2, P6) on hold, model settled as Qwen3-30B-A3B. `ops/run.py` estimator corrected: 3.34 against 30.5 billion params (*computed*, [../proofs/preregistration-p2-2026-08-14.md](../proofs/preregistration-p2-2026-08-14.md)).

## 2026-08-15. P1 measured

- `rankbench`, 2^24 blocks, prereg timestamped at 13:37 (sha256 `5109b35f`): marche-binomiale 0.3101 ns/block (kill 1.50), cascade-uniformisée 1.7809 (kill 2.00), cascade-archive 10.8115 (*measured*, [mesures/p1-rankbench-2026-08-15.txt](mesures/p1-rankbench-2026-08-15.txt)).
- Uniformizing the loop is worth an order of magnitude: 10.81 → 1.78 ns on the same bits. The walk comes in at 3.84× the floor arm.
- P5 opens (walk ≤ 0.45); the CUDA arm of P4 authorized at commit `b18fe52` (13:42:02).
- V0: 883 blocks out of 16,777,216 failed on the first cascade-archive run (*measured*, [mesures/p1-rankbench-2026-08-15.txt](mesures/p1-rankbench-2026-08-15.txt)), fixed.

## 2026-08-15 (evening). P1b, P1c, P5

- P1b: the per-block walk gives 0.6735 ns/block, ×2.17 against the ×1.002 predicted by the step count (*measured*, [mesures/p1b-marche-bloc-2026-08-15.txt](mesures/p1b-marche-bloc-2026-08-15.txt)). Green against the 1.50 kill, above the 0.45 gate.
- Authorization of the CUDA arm withdrawn at commit `c40641b` (14:39:33): 57 min (*measured*, git). "Half a day" is superseded.
- Overflow hypothesis refuted: flat arm 0.8346 against 0.6704 ns/block (*measured*, [mesures/p1b-marche-bloc-2026-08-15.txt](mesures/p1b-marche-bloc-2026-08-15.txt)). The ×2.17 stays unattributed.
- P1c: decoded E1v stream 0.6795 ns/block, addressing overhead +1.2% (*measured*, [mesures/p1c-e1v-flux-2026-08-15.txt](mesures/p1c-e1v-flux-2026-08-15.txt)).
- P5: E1v 2.3877 b/weight, transcoding 1.088× [1.087–1.090] against 2.0, 0 divisions (*measured*, [mesures/p5-cns-2026-08-15.txt](mesures/p5-cns-2026-08-15.txt)). P5 closed 4/4: right to port E1v to a card.
- Warp alignment: 0 blocks out of 150,681,600 fall in an aligned warp; padding +15.47% at 4B; aligned E1c14 5.2354 against 4.8040 (*computed*, [mesures/x3-alignement-warp-2026-08-15.txt](mesures/x3-alignement-warp-2026-08-15.txt)). E1c14 buried at 4B.
- The `.ots` for P1b and P5 are laid after the measurement (15:23): debt declared.

## 2026-08-16. E1v closed, the nullk floor

- E1v on CUDA: 0.25× [0.25–0.25], 25 GB/s, 44.253 ms, $0.85 (*measured*, [mesures/e1v-cuda-2026-08-16.txt](mesures/e1v-cuda-2026-08-16.txt)) against a floor of 1.60×. Closed for the served path.
- The format holds: 1.09 GB read against 2.18, 2.3983 b/weight in a row-aligned cut, 79 registers, 0 spill. The inline decoder multiplies the decoding term by 17 (*computed*).
- `nullk`, not a single weight byte: 2.305 ms against 5.102 for Planes14, 45.2%, 4.77× [4.74–4.77], $0.77 (*measured*, [mesures/nullk-plancher-2026-08-16.txt](mesures/nullk-plancher-2026-08-16.txt)). Planes14 buys 3.11× net; decoding ~7%.
- Written that day: "absolute ceiling of all format work = 4.77×". Refuted on 08-21.
- Aligned E1c12 4.2880 against 4.3424 for Planes12x, −1.3% (*computed*, [mesures/e1c12-aligne-2026-08-16.txt](mesures/e1c12-aligne-2026-08-16.txt)). Payload: 5.3756 · 4.6667 · 4.2029; the 08-07 table is in kernel accounting.
- E2's 1.6× criterion: priority established by the commit message `caef2ac` (10:36:27), measurement `4a09d8b` (11:28:59), without a timestamp. "No trace before the measurement" is refuted: `git log -S` does not read messages.
- The CUDA image has not compiled since 08-15 (N_ARMS 7 → 15): `arms.rs`, `bin/cuhcheck`. Lesson: make the text of a ported kernel execute (`host_e1v.cpp`, shift of 64).
- "SKIP cleanly" replaced by a named failure: eight sites were going green without the archive.

## 2026-08-17. The bucket, the paired 14B

- 14B MMLU dumps found again in the bucket: 579 kB, $0 (*measured*). "Lost, campaign to redo" is refuted. Bucket: 69 files, 46.7 GB, never inventoried (*measured*, `hf buckets ls`, [mesures/mmlupair-14b-2026-08-17.txt](mesures/mmlupair-14b-2026-08-17.txt)).
- AWQ − LLVQ at 14B: +6.09 pp [+3.62; +8.52], SE 1.25, McNemar 1.143e-11, 230/106 discordant (*measured*, [mesures/mmlupair-14b-2026-08-17.txt](mesures/mmlupair-14b-2026-08-17.txt)). Nine pairs exist.
- Drop of the MMLU gap: 4B→8B 6.96 pp, p = 0.0001; 8B→14B 1.40 pp, p = 0.40 unresolved; 4B→14B 8.36 pp (*computed*). p = 0.40 does not prove equality.
- "The gap melts twice as fast" and "it closes around 16-32B" are withdrawn.
- `rtbits` at 14B: 14,768,307,200 params; 5.106 against 5.404 for AWQ; margin −2.6 / −10.6 / −5.5% non-monotonic, mechanism = the embedding's share (*computed*, [mesures/rtbits-14b-2026-08-17.txt](mesures/rtbits-14b-2026-08-17.txt)).
- At 14B, aligned E1c14 4.6410 < 4.7063 and padding +4.18% (*computed*, [mesures/rtbits-14b-2026-08-17.txt](mesures/rtbits-14b-2026-08-17.txt)): "E1c14 buried" becomes a 4B verdict.
- Paired 8B and 14B ppl: LLVQ/f16 excess +22.01% [+19.37; +24.70] and +18.94% [+17.22; +20.68] (*computed*, [mesures/ppl-appariee-8b-14b-2026-08-17.txt](mesures/ppl-appariee-8b-14b-2026-08-17.txt)).
- Rules: `hf buckets ls`, `hf jobs logs`, `hf jobs inspect` before any quote.

## 2026-08-17 (evening). The 4B NLLs and the knee

- 4B NLLs found again in `hf jobs logs` (36 lines, sha256 `07bf4119`), $0 against the ~$0.25 quoted ([mesures/a4-campagne-4b-ppl-BRUT-2026-08-06.txt](mesures/a4-campagne-4b-ppl-BRUT-2026-08-06.txt)).
- Paired 4B excess: LLVQ/f16 +38.45% [+33.62; +43.45] (*computed*, [mesures/ppl-appariee-4b-2026-08-17.txt](mesures/ppl-appariee-4b-2026-08-17.txt)).
- Perplexity knee resolved: step 4B→8B ×0.881211, step 8B→14B ×0.974855, difference −0.100992 [−0.137670; −0.064313], t = −6.06. Melt −42.8% [−51.8; −33.5] then −13.9% [−22.8; −4.9].
- The morning's "the knee is not testable in ppl" is refuted in the evening. The "−42%" was a truncation.
- On the AWQ reference, the 8B→14B step excludes zero by 0.005 (t 2.2063 against 2.200985): never say "significantly".
- Rule: every sentence about the knee names its metric. Rule: keep the raw output.

## 2026-08-17 (second evening batch). AWQ in vLLM

- vLLM 0.26.0, L40S, batch 1, 128 tokens: f16 83.09 tok/s, AWQ Marlin 200.49 [200.39; 200.61], ×2.413 [2.412; 2.414] in-stack, $0.11 (*measured*, [mesures/awq-vllm-4b-2026-08-17.txt](mesures/awq-vllm-4b-2026-08-17.txt)).
- The vLLM control comes in at ×1.91 our dense (*computed*): a non-decomposable engine confounder. No cross-stack division; "faster than 4-bit" cannot be said at any scale.
- "No measurement against AWQ in its own engine" is lifted; the comparison ban stands.
- Forcing `awq` loads the same Marlin kernel twice (0.10% difference, *measured*, [mesures/awq-vllm-4b-2026-08-17.txt](mesures/awq-vllm-4b-2026-08-17.txt)): the clause "at M = 1 all kernels converge" stays untested.
- 8B AWQ arm blocked: two Hub revisions not validated.

## 2026-08-17 (third evening batch). The 14B served

- 14B, Planes14 + q8: 42.9 tok/s in 9.39 GB against 17.0 in 29.54, ÷3.14, ×2.53, 128 identical tokens, $1.24 (*measured*, [mesures/fusedrun-14b-2026-08-17.txt](mesures/fusedrun-14b-2026-08-17.txt)).
- The morning's "neither the speed nor the VRAM measured at 14B" is refuted.
- Cross-check: 9.39 GB × 8 / params = 5.0866 against 5.106 from `rtbits`, −0.38% (*computed*).
- The dense handicap is at its maximum here: 1,555.8 MB copied per token, 53.9 ms against 1.2 ms (*measured*, fenced profile, [mesures/fusedrun-14b-2026-08-17.txt](mesures/fusedrun-14b-2026-08-17.txt)).
- Same-head reconstructions ×1.78 and ×1.24 from the fenced profile: superseded on 08-18. "The ×2.53 is the highest of the three" is refuted by the 8B.
- `jobs.csv` register reconciled: $57.56 (*measured*, [data/jobs.csv](data/jobs.csv)).

## 2026-08-18. B2, B3, F1

- B2: medians over 5 rounds at the three sizes, ~$2.25 over three jobs (0.35 + 0.63 + 1.27; *computed*, [data/jobs.csv](data/jobs.csv); journal [mesures/b2-fusedrun-plages-2026-08-18.txt](mesures/b2-fusedrun-plages-2026-08-18.txt)). 4B q8 87.0 [86.8–87.0] in 2.56 GB, ×2.00; f16 48.3 [48.1–48.3], ×1.11 [1.11–1.11].
- 8B: 68.2 q8, 34.1 f16, ×2.57 and ×1.29. 14B: 43.3 q8, 23.9 f16, ×2.55 and ×1.41 [1.40–1.41].
- The same-head series increases: ×1.11, ×1.29, ×1.41. The raw series (×2.00 · ×2.57 · ×2.55) has no order.
- All the single points (47.0; 48.7; 88.4-88.5; 69.3; 42.9; 2.60 GB) are superseded, gaps from −1.6 to +0.9%. The "2.60 GB" was the rounded card display.
- B3: 8B resealed from the bucket, 5.322 b/param to the thousandth, $0.24 (*measured*, [mesures/b3-8b-reseal-2026-08-18.txt](mesures/b3-8b-reseal-2026-08-18.txt)) against the $12.61 provisioned.
- F1: in-house f16 control at 1.024 (2 arms) and 1.015 (5 arms) of cuBLAS, criterion ≤ 1.05, $0.08 (*measured*, [mesures/f1-cublasf16-2026-08-18.txt](mesures/f1-cublasf16-2026-08-18.txt)). Every L40S "vs FP16" holds.
- The B3 prereg's "seed 1000000" was a sentinel: erratum in the journal, prereg not edited. Rule: a timestamped prereg is never edited.
- `g6_pack`: "fails in debug, not a regression" was a real bug (shift 64), fixed in `a32163e`. The "repository without contradiction" batch: 8 catches.

## 2026-08-19. F3, F4, F5

- F3: host−device gap 0.1-0.2%, 4-8 µs per whole round, $0.86 (*measured*, [mesures/f3-events-2026-08-19.txt](mesures/f3-events-2026-08-19.txt)) against the 0.5-2 ms expected. `ncu` refused (ERR_NVGPUCTRPERM), closed. Driver 580.159.03 captured.
- F4 on A100-SXM4-80GB, ~$1.00 (*estimated*): Planes14 0.79×, Slot32 0.73×, Planes12x 0.73×, Golay70 v2 0.62×, AWQ 1.82×, cuBLAS 1.14×, nullk 1.68× (*measured*, [mesures/f4-a100-2026-08-18.txt](mesures/f4-a100-2026-08-18.txt)).
- Effective GB/s 425 → 250 and 428 → 266: bounded by compute on A100. "decode at matvec speed" becomes an L40S/Ada statement.
- F5, three full runs of the 4B, $21.45: seeds 1/2/3 at 16.7425 / 15.8836 / 15.1027. Range 10.3%, σ 5.2%, resolved pairs t +4.54 / +10.92 / +7.68 (*measured*, [mesures/f5-graines-4b-2026-08-19.txt](mesures/f5-graines-4b-2026-08-19.txt)).
- The 0.7% σ of batch B and the "noise below 1.5%" threshold are refuted at the published size. The three seeds give identical 2.0702 b/weight and 1.771 GB.
- Oracle −1.6% and volume −1.2% fall below the noise; "capped" stands. A constant-file A/B does not carry that σ.
- Day at $23.31 (*computed*).

## 2026-08-20 to 08-21. F2, QTIP on the benchmark

- QTIP in our benchmark, one process, 7 rounds of which 2 discarded, $0.89: 2.246 ms [2.245–2.248], 0.91 GB, 2.0000 b/weight, 405 GB/s, 4.89× (*measured*, [mesures/f2-p3-qtip-banc-2026-08-21.txt](mesures/f2-p3-qtip-banc-2026-08-21.txt)).
- Same process: Planes14 5.103 ms, 2.18 GB, 2.15×; nullk 2.306 ms. r = t(Planes14) ÷ t(QTIP) = 2.27× [2.27–2.28]; traffic 2.40× (*computed*).
- t(QTIP) < t(nullk): separation 2.7% against 2R = 0.72%. On 08-21, "all format work is capped at 4.77×" is refuted: nullk is the floor of our geometry.
- f = 61.1% against the 59.6% timestamped: erratum in the journal, prereg not edited.
- Mechanism: a codebook of 1.1·10¹⁴ points does not fit in a LUT, a 16-bit trellis state fits in 2 KiB; the index unfolds to 4.80 b/weight (*computed*).
- Worst error 5.4e-8·Σ|w·x| against a 1e-5 threshold. No quality claim on this arm (pseudo-random payload).

## 2026-08-23. Batch G, the clocks

- L40S 2,520 MHz, A100 1,410, pinned at max boost, ratio 1.787 ∈ [1.60; 1.95]; nullk ×1.772 (G) and ×1.781 (F4), $1.00 (*measured*, [mesures/g-horloges-planes12x-2026-08-23.txt](mesures/g-horloges-planes12x-2026-08-23.txt)).
- The ×1.78 of the A100 table is the clock ratio. That proof covers the clock alone, without an occupancy profile.
- G3: Planes12x served at 4B, 85.0 tok/s [84.7–85.1] in 2.36 GB, ×1.96 [1.95–1.96], ÷3.41, divergence at token 89, $0.79 (*measured*). Against Planes14: −2.3% of throughput, −0.20 GB.
- Planes12x stays not default by ruling: transcoding at load time 1,340 s (*measured*, [mesures/g-horloges-planes12x-2026-08-23.txt](mesures/g-horloges-planes12x-2026-08-23.txt)). "Wired is not measured" is superseded.

## 2026-08-24. TACO submission, D1

- Paper submitted to ACM TACO (TACO-2026-428) at commit `e21a8bb`, QTIP in the body. Desk reject on 08-27.
- D1, $0.24: fusion of `q+k+v` and `gate+up` by rows, 252 → 144 matvec/token, ×1.061 [1.050–1.069] within-job, band [1.00; 1.12] (*measured*, [mesures/d1-fusion-servie-2026-08-24.txt](mesures/d1-fusion-servie-2026-08-24.txt)).
- Breakdown: 87.0 → 94.9 [94.1–95.2] (rotation hoisting) → 100.6 [99.9–100.7] tok/s. The ×1.091 of the hoisting is cross-job, not publishable.
- Six criteria green: 128 identical tokens, divergence at token 89, +3,686,400 bytes exact, same NVRTC sha256 (64,776 bytes).
- Written that day: "the published tables stay at ROT_SHARE=0/FUSE=0". Lifted on 08-31.
- The project's front is now the launch geometry, the one `nullk` measures.

## 2026-08-25 to 08-27. Gain bits, MMLU noise, timestamps, Zenodo

- The repository stays public during review (08-25). The "private repository" of the submission note is superseded.
- Gain bits, 0.6B/28 blocks, iso-rate 2.1656 b/weight, 86 min of Mac time: leech0c13 39.3309, leech2c11 39.5350, leech1c12 43.4865, leech4c10 47.1537 (*measured*, [mesures/gain-ab-gate-0.6b-2026-08-25.txt](mesures/gain-ab-gate-0.6b-2026-08-25.txt)).
- The gain-bit ladder is refuted: seed 1 reverses the ranking, one arm moves by 13.9% against 10.6% of spread between the four. Radial bias +3.69% (*measured*, [mesures/cosdiag-biais-radial-0.6b-2026-08-25.txt](mesures/cosdiag-biais-radial-0.6b-2026-08-25.txt)).
- Cross-seed MMLU noise at 4B: 58.02 / 52.19 / 55.17, s = 2.92 pp, $0.58 (*measured*, [mesures/bruit-mmlu-graines-4b-2026-08-25.txt](mesures/bruit-mmlu-graines-4b-2026-08-25.txt)). The 0.5-1.5 pp prediction is refuted; the volume ladder was not launched, ~$19 saved (*estimated*).
- 08-26: 16 of the 20 `.ots` carry 3-4 Bitcoin anchors (*measured*, [mesures/ots-etat-2026-08-26.txt](mesures/ots-etat-2026-08-26.txt)). The "0 anchors, 4 pending" of 08-25 is refuted: grep was blind to an 8-byte binary tag.
- Preregs of 08-10 and 08-11: the anonymization pass (`01fdbe6`) rewrote their bytes; none of the 128 git blobs yields the digest. That debt is declared in the paper.
- 08-27: TACO desk reject on scope; `ots upgrade` 20/20 anchored ([mesures/ots-etat-2026-08-27.txt](mesures/ots-etat-2026-08-27.txt)); Zenodo concept DOI 10.5281/zenodo.22133606.
- Closing plan: 9 batches, $9 to $13 (*estimated*). The evening handover quoted $0.49-0.55 for a job that had already succeeded: fifth catch of the retention rule.

## 2026-08-28 to 08-30. Post-deposit plan, isolated stacks

- Post-deposit plan (08-29): freeze ~$0.25, geometry ~$2-4, quality ~$12-25, families ~$17, MoE ~$65 (*estimated*). The "Hessian calibration" mini-paper is buried. Outreach drafts written, not published.
- Phase P (vLLM port before the geometry) laid down that evening; reversed on 08-31 (`deaa449`).
- First M3 gate red on us: macro aggregate 72.85; in micro 70.36; f16 across four engines [70.3; 70.9] (*measured*, [mesures/m3-gate-mmlu-vllm-2026-08-30.txt](mesures/m3-gate-mmlu-vllm-2026-08-30.txt)); second gate 70.34 (*measured*, [mesures/m3-gate2-mmlu-vllm-2026-08-30.txt](mesures/m3-gate2-mmlu-vllm-2026-08-30.txt)).
- IQ2_XXS on Metal: 2.0625 bpw, ×2.6287, MMLU 39.39; LLVQ − IQ2_XXS +16.20 pp [+12.64; +19.72], reading threshold ~6 pp (*measured*, [mesures/m3-iq2-metal-2026-08-30.txt](mesures/m3-iq2-metal-2026-08-30.txt)). Served 2.479 against 5.162 b/param.
- Same GGUF on CUDA: ×3.688, MMLU 38.87, 96 disagreements (*measured*, [mesures/m4-iq2-cuda-2026-08-30.txt](mesures/m4-iq2-cuda-2026-08-30.txt)). llama.cpp f16 84.83 tok/s, vLLM 83.09: agreement 2.1%.
- GPTQ 2-bit: artifact 1,754,463,312 bytes, 3.489 b/param (*measured*, [mesures/m3-gptq2-production-2026-08-30.txt](mesures/m3-gptq2-production-2026-08-30.txt)); the "3.182" on the gptqmodel denominator is superseded. MMLU 24.74 degenerate, not publishable.
- M3/M4 campaign: $1.29 over 11 rows (*measured*, [data/jobs.csv](data/jobs.csv)); the protocol counts 12, the discrepancy is not explained.

## 2026-08-31. Wave 2, v1 freeze

- Fusion at the three sizes: ×1.055 [1.054–1.058] at 8B, ×1.028 [1.027–1.029] at 14B, band [1.00; 1.12]; overheads +4,423,680 and +6,717,440 bytes exact (*measured*, [mesures/vague2-fusion-8b-14b-2026-08-31.txt](mesures/vague2-fusion-8b-14b-2026-08-31.txt)).
- Served config v1 frozen: Planes14 + q8 + ROT_SHARE=1 + FUSE=1, 100.6 / 75.5 / 46.8 tok/s in 2.57 / 5.41 / 9.40 GB. Rule written before the numbers.
- The ban "an isolated fused 4B would break the property" is lifted by the freeze. The same-head series is not re-measured under v1.
- Prereg committed 77 s before the job was created (*measured*, git). Space in BUILD_ERROR for ~9 h 40 min (*measured*, [../proofs/preregistration-vague2-gel-geometrie-2026-08-31-ECARTS.md](../proofs/preregistration-vague2-gel-geometrie-2026-08-31-ECARTS.md) §É2).
- Operator decision `deaa449`: A2 and A3 before the vLLM port. Isolated-stacks protocol v2 timestamped, constants anchored: 4,022,468,096 params (*measured*, four instruments), f16 standard [70.3; 70.9] ([../proofs/protocole-piles-isolees-v2-2026-08-31.md](../proofs/protocole-piles-isolees-v2-2026-08-31.md)).
- Adversarial verification of the v1 alignment: 25 agents, 7 surfaces.

## 2026-08-31 (evening). A1, A4

- A1: nullk 144 against 252 launches, 1.794 against 2.200 ms, r = 0.8158 [0.8150–0.8162] (*measured*, [mesures/a1-nullk-252-144-2026-08-31.txt](mesures/a1-nullk-252-144-2026-08-31.txt)); 3.76 µs/launch (*computed*, 0.406 ms over 108 launches). Prior 0.83 confirmed to 1.7%.
- A1 died four times before returning a number, three times from infrastructure and once from the launcher, for $0.02; each death is in the deviations file ([../proofs/preregistration-vague2-gel-geometrie-2026-08-31-ECARTS.md](../proofs/preregistration-vague2-gel-geometrie-2026-08-31-ECARTS.md)).
- r falls in the mixed band, between the 0.65 and 0.90 thresholds. The A2/A3 order goes back to the operator.
- A4 on A100: r = 0.8198, times stretched ×1.809 (clocks 1.787); fusion ×1.063; fused 63.4 against dense 51.4 tok/s, $0.83 (*measured*, [mesures/a4-a100-2026-08-31.txt](mesures/a4-a100-2026-08-31.txt)). F4 reproduced (0.79×, 0.73×, 1.14×, 1.69×).
- Wave 2 complete: $2.17 against a $5 cap (*measured*, [data/jobs.csv](data/jobs.csv)).

## 2026-09-01. A2/A3 prereg

- Ruling: A2 (CUDA Graphs) first, commit `833d630`, prereg sha256 `802006c5` timestamped before any job ([../proofs/preregistration-a2-a3-geometrie-2026-08-31.md](../proofs/preregistration-a2-a3-geometrie-2026-08-31.md)).
- Per-launch pool extrapolated to 252: 0.947 ms ≈ 43% of the floor, linearity declared (*computed*).
- Thresholds: adoption ≥ 8% end-to-end, closure < 3%, A3 benchmark gate ≥ 10%, phase kill < 8% cumulative, $4 cap.
- Declared priors are unfavourable: CUDA Graphs closed in batch A at 0.167 ms = 0.8% of a token (*measured*, batch A), reopened by decision. KV preallocation dev: 2-4 days (*estimated*).

## 2026-09-01 to 09-02. A2 and A3 delivered

- A2 step 1: prealloc/cat 0.8919 [0.8884–0.8953], prior 1.00 refuted; extended store 0.9917 [0.9883–0.9935] (*measured*, [mesures/a2-verdict-2026-09-01.txt](mesures/a2-verdict-2026-09-01.txt)).
- A2 hybrid graph at 4B: 99.2 → 112.5 tok/s [112.4–112.6], +13.45% [13.36–13.58]; 8B +10.1%; 14B +6.1% (*measured*, [mesures/a2-transfert-verdict-2026-09-01.txt](mesures/a2-transfert-verdict-2026-09-01.txt)), $0.87. Adopted on the criterion; curve point at 14B; no v2 freeze.
- A3, eight occupancy variants, 1,105,920 bit-exact rows (*measured*, [mesures/a3-occupation-banc-2026-09-01.txt](mesures/a3-occupation-banc-2026-09-01.txt)). pers gives +1.56% [+1.01; +1.86], below the gate. persall gives +26.36% [+25.31; +26.61], a benchmark arm that does not port.
- Split-K sk1 gives −1.87%: "the underfill of o/down is the residue" is refuted. Phase kill not triggered. Phase A: $1.11 of $4.
- 09-02, operator decision: A2 is not served. 8k KV window: +1.21 GB on 2.57, +47% of VRAM for +12.6% of throughput (*computed*, never measured, [../proofs/preregistration-a2-a3-geometrie-2026-08-31-ECARTS.md](../proofs/preregistration-a2-a3-geometrie-2026-08-31-ECARTS.md) §É7). At 8B +22%, at 14B +14%; at 2k +12%.
- The only window that ran: prealloc(256), 0.038 GB (*computed*, ECARTS §É7); the −0.83% of é1b is a cost in time (*measured*, [mesures/a2-verdict-2026-09-01.txt](mesures/a2-verdict-2026-09-01.txt)). `KvStore::Cat` stays the default; `LLVQ_KV_PREALLOC` and `LLVQ_GRAPH_AB` are measurement modes.
- Counters: 102 jobs for $92.51, 28 `.ots` of which 20 anchored (*measured*, [mesures/ots-etat-2026-09-02.txt](mesures/ots-etat-2026-09-02.txt)). The "89 jobs / $90.55" of 08-31 is superseded.

## 2026-09-02. D0, the research roadmap

- Research roadmap adopted in three OKs, merged into `main` (`1e8583c`), $5 cap for wave 1, M1 in parallel on the Mac.
- M2 goes ahead of M1: constant-file A/B, bar 0.43 pp, quote ≈ $2.3 (*estimated*, $0.19/arm). Q5 opens if the target is `k` (+0.05 b/weight); a `down` target would cost ≥ +0.49 (*computed*).
- Plumbing checked on the Mac: k_proj 94,371,840 weights, "all restored" = checkpoint at 114/114 picks (*measured*, [mesures/m2-plomberie-mac-2026-09-02.txt](mesures/m2-plomberie-mac-2026-09-02.txt)).
- Shipped: `LLVQ_RESTORE_F16=<types>|all` in `bin/mmlu` and `bin/ppl` (requires `LLVQ_MODEL`, rejects an unknown value); `LLVQ_H_SHRINK=ρ` in `bin/smoke`. Branch `recherche/m1-m2-vague1`, `main` = `origin/main`.

## 2026-09-02 (continued). M2, M2b, M1

- M2, job `6a97ea8e`, 72 min, $2.17: 11 arms, controls 55.59 and 70.32 at 2280/2280 picks (*measured*, [mesures/m2-attribution-4b-2026-09-02.txt](mesures/m2-attribution-4b-2026-09-02.txt)). The $2.3 quote is superseded.
- Paired gains by restored type, in pp of MMLU (*measured*, same journal):

  | restored | gain [CI95] |
  |---|---|
  | gate | +5.18 [3.04; 7.34] |
  | up | +4.94 [2.72; 7.17] |
  | v | +4.48 [2.39; 6.61] |
  | down | +2.96 [0.71; 5.17] |
  | o | +2.35 |
  | k | +2.09 |
  | q | +1.85 |
  | attention | +6.90 |
  | MLP | +10.78 |
  | all | +14.73 |

- The literature prior (k_proj, attention) is refuted. Target v_proj: 2.6% of the weights, yield 8× the best MLP target (*computed*).
- Deviation É1: v_proj in f16 = +0.263 b/param (5.425 > AWQ 5.302); in int4 g128 = −0.013 (5.149) (*computed*). Cause: Planes14 unfolds to 4.804 b/weight for 2.07 of information.
- M2b, job `6a986698`, 10 min, $0.29: v_proj dequantized from int4 g128 gives MMLU 59.19, +3.60 [1.47; 5.79], McNemar 2.0e-4, 80.4% of the f16 gain (*measured*, [mesures/m2b-v4bits-2026-09-02.txt](mesures/m2b-v4bits-2026-09-02.txt)).
- The prereg rule has a hole: line 1 requires CI > 1.5 (bound 1.47), lines 2-3 require G4 < 3.0. Arbitrated on 09-04 ([../proofs/preregistration-m2b-v4bits-2026-09-02-ECARTS.md](../proofs/preregistration-m2b-v4bits-2026-09-02-ECARTS.md)).
- M1, $0, 12 Mac runs, 0.6B/28 blocks, median and range over 3 seeds: ρ = 1 39.6042 / 4.6214; 0.9 27.0812 / 3.1498; 0.7 27.4944 / 0.6847; 0.5 27.9506 / 2.9771 (*measured*, [mesures/m1-hessienne-shrink-2026-09-02.txt](mesures/m1-hessienne-shrink-2026-09-02.txt)). Control 38.4507 replayed.
- By the prereg rule: ρ* = 0.7, M1 green, the signed kill prediction (ρ* = 1) refuted. On n = 3 the range hangs on one seed; the sign and the order of magnitude hold (−12 ppl, seeds 2-3 from 3.47 down to ≤ 0.54). Q1 adopts ρ ∈ [0.5; 0.9], to be re-estimated; n/N 0.023 against 0.074 at 4B (*computed*).
- M1 deviation: the queue moved to nice 10 at the 5th measurement (CPU 1470%, RSS 1.22 GB), ppl bit-exact; rule `LLVQ_THREADS ≈ ncpu−4` and nice from launch. F1 note: projected retention 88.9-89.6% below the 90.3 kill (*estimated*), F1a counts before coding.
- arXiv submission 7927047 rejected: `paper.pdf` uploaded in place of the sources; `\pdfoutput=1` added to `main.tex` for resubmission (`e721bc5`, git). Preregs timestamped: m2-attribution `71712e60`, m1-hessienne-shrink `5a5e1027`, m2b-v4bits `263ec52a`, anchoring pending. Wave 1: $2.46 spent of $5.

## 2026-09-04. M2b arbitrated, F1a counted, and the rule that a gate names a fundamental criterion

- Operator decision: the case the M2b rule left uncovered is read cashable. Both axes move at once, +3.60 pp of paired MMLU for −0.013 b/param, and nothing is paid for the gain. Q5 opens. The decision does not repair the rule: line 1 stays failed on its letter, the CI lower bound reaching 1.47 and never 1.50 over eight bootstrap seeds (*measured*, [m2b-v4bits](mesures/m2b-v4bits-2026-09-02.txt), [ECARTS](../proofs/preregistration-m2b-v4bits-2026-09-02-ECARTS.md) §É4).
- M2 replayed on a second draw, $2.14: nothing between the two draws is resolved, the largest difference being `gate` at z = −1.68, and the head of the ranking swaps, `gate` from 1st to 4th and `down` from 4th to 1st (*measured*, [m2rep-graine3](mesures/m2rep-graine3-4b-2026-09-04.txt)). Stable across draws: MLP above attention, `v` third, `q` and `k` last. The prereg's signed prediction is refuted on the published file and confirmed on seed 3, which is draw dependence, measured.
- M2b on the same draw, $0.45: `v_proj` in int4 g128 gives +2.71 pp [+0.59, +4.93], p = 0.0106 (*measured*, [m2b-graine3](mesures/m2b-graine3-4b-2026-09-04.txt)). The CI clears zero, so Q5 is adopted and the mixed-precision kernel is built. **The served figure is the range +2.71 to +3.60 pp over two draws, never +3.60 alone.** The survival rate of the f16 gain into int4 is 94.4% here against 80.4% on the published file, so it is not a constant of the format.
- Three independent jobs in three weeks return a dump byte-identical to `mmlu-s3.csv`. Wave 1 closes at $5.05 on a $5.00 cap, declared.
- F1a, first exact count, $0: a shaping region that is the product of three 8-dimensional balls gives **89.10%** of retention at 2.000 b/dim against a kill at 90.3% (*computed*, `ops/f1a_shaping.py`). The denominator trap was avoided: the word is 48 bits for 24 dimensions on both sides, where dividing by the 47 bits of the lattice-point field alone would give 90.99% and pass the kill.
- **The decisive result is a ceiling, not a measurement.** Any per-section region is a product of three 8-dimensional regions and the best of those is the ball, so no rule of the form `[state][s1][s2][s3]` exceeds 0.7292 dB, where the F1b kill needs 0.8742. F1b cannot pass its own gate whatever the truncation rule; only joint bounding across sections can, and that is F2 (*computed*, nine agents, four families adversarially attacked, [f1-regle-de-troncature](archive/f1-regle-de-troncature-2026-09-04.md)).
- Voronoi codes, predicted here as the answer, are dead on decode cost: about 1,018 operations against a 250 budget, the E8 nearest point alone costing 217 per section. The prediction was right on bijectivity and wrong by an order of magnitude on the decode.
- The branch count of the middle section was wrong by a factor of four, found by an adversarial review: 1,024 edges over 64 Golay states is 16 branches per state, not 4 over 256 Λ₂₄ states. The 8.0 KiB table and its "green with 2× margin" are withdrawn. The closing check had passed because the two errors cancel exactly, 512 × 4 = 128 × 16. A check that cannot tell the right count from the wrong one is not a check.
- The decoder table computed for the served split: under the only isometry the kernel applies for free, the 256 end-section regions fall into **67 orbits** and the 16 middle ones into **9**, which is **3,336 KiB, 34 times over** the card's opt-in (*computed*, `examples/f1table.rs`, reproducing an adversarial review from an independent implementation). The consequence is traffic, not capacity: 454 M lookups a model pass is 14.5 GB of table reads against 0.98 GB of weight reads, fifteen to one, where `Planes14` decodes from a 12 KiB L1-resident table.
- **Operator rule, and it outlives F1: a gate is written on a fundamental criterion, never on a proxy.** The four axes, plus loadable model class, encoding cost and noise floor. It came from the day's audit: all three unsound F1 gates were proxies (Gaussian retention, table kibibytes, a competitor's milliseconds in its own grid), while the Q and M axes, already written on perplexity, MMLU and b/param, survived untouched. Recorded in [METHODE.md](METHODE.md) §1. The F axis loses two decision points and its first gate becomes F1c, on perplexity.
- F1b built on the Mac, $0, and mutation-tested rather than trusted after two checks passed on wrong values the same day: ten mutants tried, nine killed, the survivor (a k-range short by four, invisible to a brute-force comparison at small `t_max`) closed with an always-on range guard, and three confirmed equivalent by enumeration. Walking the trellis rebuilds the permuted Golay code set for set, and the truncation radii match, bound for bound, what an independent Python implementation derives from the same construction.
- The section encoder's bias was measured, not assumed: exact on 400 of 400 targets in each section against exhaustive search, and one-sided against F1 where it is not. Throughput 1.473 to 0.122 s a block after hoisting the section-3 search, with a mean `t` identical to four decimals.

## 2026-09-05. The table floor, and a $0.01 lesson

- The CUDA image built in 19 min against the 40-70 estimated, and the 399 lines of `f1floorbench` that macOS cannot type-check compiled on the first try, the hand audit against `nullkbench` had found the only two errors it had.
- The job then died in seconds, **$0.01**, on `no embedded copy of f1floor.cu`. `load_sources_many` reads an `include_str!`'d table and the new unit had no arm in it. Invisible to the type checker, invisible to `cuhcheck` (which parsed the file from disk and never asked whether the binary carried it), and invisible to any Mac because the table was gated on Linux.
- Both halves closed rather than the one that bit: the embedded constants and the lookup are **un-gated**, so any platform can see the table, and `bin/cuhcheck` now asserts that every table-shipped unit has its arm *and* is parse-checked here. The guard was verified by removing the arm again, it fires, exit 1.
- Its reverse check found a pre-existing hole in passing: `preflight.cu` was shipped by the table and had never been syntax-checked. It parses, and it is now in the list.
- Second attempt, and the bench's own guard fired for **$0.00**: the L40S carries **96 MiB of L2**, not the 48 MB the design assumed, so the 1 GiB DRAM calibration point was only 10.7× the cache, below the 20× the prereg demands, where a "miss" is a tenth hit. Moved to 4 GiB, 42.7×, and allocated zeroed on the device rather than uploaded, because 4 GiB of host RAM in a container of unknown size is a worse bet than contents that are never read for meaning.
- Two failures, $0.01 total, both caught by something written to catch them. The measurement has not run yet.
- Third attempt, **$0.01**, 18 s of card: D(8 KiB) 0.344 ms, D(16 KiB) 0.663, a flat plateau of 4.52 ms from 128 KiB to 16 MiB, 10.6 at 64 MiB, 61.4 at 4 GiB; Dsm 3.33 / 5.10 ms; Didx −0.103 ms, negative and unexplained (*measured*, [f1-plancher-table-2026-09-05](mesures/f1-plancher-table-2026-09-05.txt)). Signed prediction right on D(4 MiB) and D(16 KiB), wrong on Dsm(48 KiB) (0.5–2 predicted).
- Reported the same morning as "F1 is dead on decode cost, ~3× the budget, and placing the hot set in shared memory makes it worse". **Withdrawn the same day** after an adversarial audit (six independent readings, three counter-verified, four $0 computations on the Mac, 800 to 27,376 blocks encoded). What fell: the bench runs at six blocks per SM, not eight, so L1 is 28 KB with 12 KiB of tile in it and the 16 KiB point already misses; the shared-memory arms confound occupancy and 3.4–6.8 GB of per-block staging with placement, and Dsm compares 8 warps against 48, the cross-occupancy reading §6 forbids; "worse" is withdrawn (QTIP does 1.82 G shared lookups per pass in 2.246 ms in the same F2 bench). What held: the plateau is a sector throughput, 3.2 TB/s, not a latency; the real access distribution, computed for the first time, puts 17–21% of lookups in the hottest 16 KiB against the 71–77% the budget needs, so the 67/9 table sits at 2.4–2.5× H, a fact about that table.
- Corrections in passing: 452,044,800 lookups per pass, not 454,164,480 (tails); an entry fits in **4 bytes** for every orbit ((y − p)/2 ∈ [−5, 4]), so the table is 2,224 KiB, not 3,336, `f1table.rs` counted the sign twice; the L40S's L2 is 96 MiB (the 09-04 entry above says 48 MB).
- **Operator rule, 2026-09-05: a kill is written on a fundamental criterion and by the operator alone.** A floor, a bracket or a projection informs it and never pronounces it. Recorded in [METHODE.md](METHODE.md) §1; deviations of the floor prereg in [ECARTS](../proofs/preregistration-f1-plancher-table-2026-09-04-ECARTS.md).
- Out of the audit, a decoder that fits the floor: a **universal rank table**, one 16 KiB table (2 parity classes × 2,048 rank vectors × 4 B) for all 528 regions, exact for p = 1 and a two-profile compromise for p = 0. Measured on the F1b harness, same blocks, same process: 89.05% against 89.69% for exact F1 (2,000 blocks, −0.64 pp), then 88.88% against 89.48% (4,000 blocks, −0.61 pp), ball-12 control 92.00% both times, paired MSE loss 1.7 ± 0.2%, 0 blocks outside Λ₂₄ (*measured*, `examples/f1rankbench.rs`). The signed prediction before the run was a loss of 2.0–4.5 pp. Its table costs 0.34–0.66 ms on the measured curve; the unknown is now the arithmetic decode, where E1v died.
- Also found by the audit, on fundamental criteria the floor never touched: the bench encoder runs at 240 ms/block/core against F1c's 656 µs/block gate, **366×**, a production encoder precedes any F1c; F1e's "≤ 2.6 b/param" was unreachable at 4B by construction (q8 embedding: 2.76), rewritten on the triplet's b_max; under F1 the 70B fits the triplet's 27.93 GB at 19.5 GB (*computed*) while the `rot_apply` wall still closes the served path past 14B.
- The F1b journal, cited by the floor prereg, did not exist: written on the 05 from the raw output kept in the session transcript ([f1b-retention-2026-09-04](mesures/f1b-retention-2026-09-04.txt)), with its deviations (control §4.5 replaced by `f1enc` + a one-sided argument; two splits on 2,000 blocks; a wrong-grid pilot at 64.65%).
- **The universal-table decoder compiled, verified on the card, and measured** ($0.01, 17 s; prereg stamped before, sha256 `119b02d5…`; [journal](mesures/f1-rang-plancher-2026-09-05.txt), [ECARTS](../proofs/preregistration-f1-rang-plancher-2026-09-05-ECARTS.md)). Built by two agents against one spec and reviewed by three (decoder equivalence: 44 mutants on the header killed by the clang++ harness; compile-ability: the Linux half type-checked on the Mac for the first time with `CUDARC_CUDA_VERSION=12040 --target x86_64-unknown-linux-gnu`, now in CLAUDE.md; protocol and registration). On the card: 64,512 blocks decode to the Rust reference's coordinates; 48 registers, 0 local; `T = 3.346 ms` against `B = 2.797`, 1.20×. The signed prediction was wrong three times (S below: 0.645 for 0.9–1.3, a difference not a bandwidth; Du above: 2.701 for 0.9–1.6; T above: 3.346 for 2.0–3.0) and right on registers. The table is 0.66 of Du; the arithmetic and the dependent small-table chain ~2.0 ms, kernel work, not table work. The prereg's row: F1d is written with this decoder, ~95 tok/s projected at the 4B (*estimated*) for 1.36 GB, and the operator weighs it.
- **The production encoder prototyped** ($0, [journal](mesures/f1-encodeur-prototype-2026-09-05.txt)): 99.7% of the bench's 240 ms were 191,488 membership tests per (block, scale); membership in closed form (cost < C or a lexicographic cut, checked over all 8⁸ rank vectors), the bench's rule kept to the letter (307,200 solves, 0 disagreement), and a lazy trellis encoder: **290–296 µs/block/core** in three runs against the 656 µs gate, the bench's points on 6,000 (block, scale) pairs. Scales measured: one costs 1 pp, two adaptive 0.16 pp, three sit at the gate. Its counter-review: the gate is an encoder-only figure (`encbench` 680–709 today), the pruning rate is unmeasured on real residues (exhaustive: 962 µs), α was fixed on evaluation blocks, and class = [cost ≡ 0 mod 16].
- **The format-v2 map drawn and counter-reviewed** ([ROADMAP](ROADMAP.md) §2.2 ter): three crates and the `llvq-llm` wiring, no shader; found on the way, the disk bit order is MSB-first where the F1 word is little-endian (a transcoder, pinned by test), `read_matrix_raw`'s signature has 20 callers (a per-file code kind in the header instead), and **F1c's gate as written has no power at ρ = 1**: the ±1 cross-seed range is ±11.7% of the median while the expected effect is +0.7 to +1.8 ppl (*estimated* from +8.5% of MSE). A paired Δ per seed at a fixed ρ with a signed prediction is proposed; the gate's form is the operator's.
- **The encoder on real blocks, $0** ([journal](mesures/f1-encodeur-blocs-reels-2026-09-05.txt)): 20,000 compensated, rotated blocks of Qwen3-0.6B captured from the served GPTQ loop by a recording `BlockQuantizer` (`llvq-llm/examples/f1recdump.rs`, deterministic dump, sha256 identical on two captures), **298 µs/block/core**, ratio 1.00 to Gaussian, base-member rate 55.5% against 55.6%, no drift by matrix or column position; retention gap to ball-12 identical (−3.20 against −3.18 pp). The rotation makes the blocks Gaussian (kurtosis 3.01). The encoder's load-bearing assumption is closed.
- **Three arithmetics for the same decoder**, written independently against one spec, host-verified, one bench ($0 so far): V1 builds the float without the int→float pipe and runs `tv_f1r`'s exact FMA chain; V2 proved the three trellis byte maps **linear over F₂ under the current numbering** (k = 0 on all 4,096 paths, 12 mask columns, `examples/f1linear.rs`, pinned by four tests in `f1::rank`) and computes the patterns with no load, so the format does not move; V3 reads the values as bytes through PRMT. Twenty-eight mutants across the three, 26 killed, 2 equivalent. The review caught a real one before any card time: the CUDA intrinsic `__byte_perm` masks its selector to three bits per nibble, so the `prmt` sign-replication mode V3's byte mask relied on is unreachable, it would have decoded wrong on the card while passing the host shim, which models the instruction. The multiply-mask form is now the default. Control 7 of the bench is the prereg's per-row form; a first draft justified an ∞-norm form with a synthetic model that put 539 rows over the tolerance, replayed with the bench's exact arithmetic, the worst row drifts 3.3e-7, and the draft was wrong.
- **The three arithmetics on the card** ($0.00, 4 s of card after 50 min of queue; prereg stamped, sha256 `a3765c0a…`; [journal](mesures/f1-rang-variantes-2026-09-05.txt), [ECARTS](../proofs/preregistration-f1-rang-variantes-2026-09-05-ECARTS.md)): every variant equal to `tv_f1r` on all 30,720 rows (V1 at Δ = 0 exactly, V2 and V3 at 3.16e-7, the review's host replay had said 3.3e-7), all three at 40 registers and 0 local. **V3: T = 1.694 ms against B = 2.797, 0.61×; V1 1.719; V2 3.444 (+0.11 on `tv_f1r`).** The 24 int→float conversions per block were ~1.6 of the ~2.0 ms of arithmetic measured at 14:53; the dependent chain of small-table reads cost nothing. The signed prediction was wrong on all four times, in the direction its own instructive clause had named (`Du_v3 < 1.3`). Reproducibility of the reference arms across the two jobs: 0.4%. The prereg's row: F1d takes v3, v1 equivalent within the ±0.1 ms resolution. Projected, same reserves as the morning's floor: ≈ 113 tok/s at the 4B, +12%, for half the VRAM (*estimated*), where the naive writing of the same decoder projected −5% five hours earlier.
- **Trio, steps 0, 1 and 3, built and reviewed the same night** ($0; three builders, three adversarial reviewers; commit `cc23f9a`). Step 0: `llvq_search::trio`, 1,122 lines, rebuilt from `llvq_core::Golay` with its own construction and asserting at construction the trio, the 64 states, the 1,024 edges, N0 = 1,240, the closed-form bounds and the 12 F₂ columns; `decode` in natural order, `encode` the inverse refusing every non-codeword; held to the bench's decoder on 10⁵ words, 10⁶ round trips, 16 mutants killed. Step 1: the production encoder, allocation-free and `Sync`, the bench's rule to the letter (6,000 of 6,000 pairs at the bench's point), α = 0.3218 and the pair (α, 1.14 α) fixed on the 4,000 training blocks before the evaluation blocks were read once (88.89% against 92.00); **329 µs/block/core** against `nearest_angular`'s 687 in the same process (*measured*, `trioencbench`, reproduced by the reviewer at 328 and 327), gate 656; the truncated rule stays a knob at 246. Step 3: format v5 (`LVQ5`), the kind in the header, the Trio fingerprint beside the untouched v1 one, encode/decode by kind, refusals in every runtime transcoder, in the fused loader and in every tool that would read a Trio index as a Ball class, seven cuda/metal bins the review found unguarded included; the default writer still writes v4 byte for byte. The review's other finding: the per-file kind cannot carry Q5's `v_proj` in int4 beside Trio matrices; a per-matrix kind goes in before any Trio file is sealed.
- Kept from the audit's ten example files, four: `f1accesscv` (real access, held-out), `f1shrink` (orbit counts under signed permutations 67/9 → 6/3, second moments, six-section trellis), `f1rankbench` and `f1rankenc` (the universal table and its encoder's exactness). Operator go, 2026-09-05: journals, the ALU floor of the universal decoder at ≤ $0.10, and the objective **an F1 that can be tested**.
## 2026-09-06. Tetra: the format is named, written, and writes its first file

- The format of lead F1 is named **Tetra** (operator, 2026-09-06): the word has four fields, `[state][s₁][s₂][s₃]`. The entries above say Trio because that is the name the work was done under. The **trio** keeps its own name where it means the mathematics: three disjoint octads of the Golay code, the standard term, `llvq_search::tetra::TRIO`. The rename moved the fingerprint's domain string and with it `PUBLISHED_TETRA_FINGERPRINT`, from `0xebc7_5263_8c8d_b088` to `0x9c30_6008_6c7a_13e6`; it was done the day the first artifact was written, which is the only day it is free.
- Steps 2 and 4 of the plan ([ROADMAP](ROADMAP.md) §2.2 quater): `TetraShapeGain` implements `BlockQuantizer` with the served recipe unchanged, and `smoke`, `seal`, `ppl`, `mmlu` and `export` read and write the format. One departure from `LeechShapeGain`, deliberate: `reproject` re-encodes instead of negating the point, because **the Tetra map is not centrally symmetric**, negating a coordinate swaps its rank inside its pair, and rank 7 has no negative at all. Measured: the map refuses the negation of 106 of 400 Gaussian blocks, all of even parity (*measured*, `llvq-quant/tests/g5_design_c.rs`).
- A code kind **per matrix** landed before any file was sealed, so Q5 can serve `v_proj` in int4 beside Tetra matrices later: each v5 record carries its own kind, the header keeps a default and a `kinds_present` mask, and every reader consults the record.
- **The first Tetra artifact exists** (*measured*, Qwen3-0.6B, 3 blocks, one seed, same corpus, one variable): witness `leech1c12` ppl 20.7935 against Tetra 21.4947, both at an effective rate of **2.1656 b/weight to the fourth decimal**, both verified bit for bit over 47,185,920 weights, both sealed and reopened at exactly their smoke perplexity. The sealed files differ by 100 bytes: 16 of header plus 21 records times 4 of tag. Tetra encodes at **42 s/block against the witness's 64**. The served 4B still reads 16.9415. Three blocks and one seed is a plumbing test, not a quality reading; that is step 5.
- Two review findings fixed the same day. `verify_artifact` proved the weights and never the label: a mislabelled file decoded against its own tag and passed, and **48.9% of Tetra words decode to a point the ball indexer accepts**, so the writer's refusal was a coin flip per block. It now reads the raw record's kind and holds it to the codebook the run was asked for. And `TetraShapeGain::reconstruct` was covered by nothing, two mutants survived the whole workspace, so it is pinned against an independent formula, which kills both.

## 2026-09-06 (continued). Tetra at the 4B: half the memory, better perplexity

- **The 4B exists in Tetra and is measured** ($0.79, prereg stamped before the first measurement; [journal](mesures/tetra-4b-2026-09-06.txt), [deviations](../proofs/preregistration-tetra-4b-2026-09-06-ECARTS.md)). Encoded on the Mac in **2 h 27** against the published run's 4 h 01, 0.981 GB at 2.1595 b/weight of projections: the published rate to the fourth decimal, every one of 3,633,315,840 weights read back bit for bit, sealed at 1,770,529,149 bytes against 1,770,527,533.
- The quality campaign is four arms on one L40S, one harness, the same token fingerprints: f16 12.2369 / AWQ 13.5207 / `Planes14` 16.9422 / **`Tetra` 16.1569** in perplexity, and 70.32 / 70.04 / 55.59 / **53.49** in MMLU micro. The two reference arms replay their A4 values of 2026-08-06 to the hundredth, which is what makes the two new lines readable.
- **Tetra against the served format: 2.7645 b/param against 5.1619 (÷1.867), the same disk, a perplexity better by 4.64% and an MMLU lower by 2.10 pp**: an interval the calibration draw alone spans (2.92 pp). In excess log-likelihood, the only cross-paper comparison [fiche-4b](fiche-4b.md) holds valid: 0.2779 nats against 0.3254 for `Planes14` and 0.3171 for QTIP, so **12.4% better than QTIP** where the repository had been 2.6% worse.
- The signed prediction was **wrong on the sign of perplexity** (+1.5 to +4% predicted, −4.64% measured) and right on MMLU to 0.9 pp. Its own instructive clause named the case: Gaussian retention overstates the loss on real weights: 88.89% against the ball-12 control's 92.00 implied +8.5% of MSE, and perplexity did the opposite.
- **Control 0 failed, and it matters beyond this measurement**: one block re-encoded under `leech1c12` today does not reproduce the published file's block 0: tail, gains and 87% of indices differ (82 to 91% by matrix) while dimensions, rotation seed, gain centroids and row scales are identical. The tail is weights kept exact after GPTQ compensation and it differs by 42 to 56% of their magnitude, so the Hessians differ and neither the rotation nor the quantizer does. Most likely cause, *not proved*: commit `4a3e5f0` of 2026-08-26 changed the calibration volume from 8,000,000 characters to `n_calib × calib_len × 6`. **The repository no longer reproduces its own published artifact**, so A4's numbers compare to nothing encoded after that date. The evaluation harness is intact. Operator's decision: proceed without re-encoding a witness, so the gap to `Planes14` holds the format and the drift together.
- Cost of the day: $0.79, of which $0.01 on a job that died in 16 seconds because the measurement image carries the Rust binaries and not the `hf` CLI. Both sealed files went through the bucket instead. Wave 2 stands at $0.82 of $2.00.
- What is left of the plan: step 6 alone, the served kernel `tv_tetra48` and the comparison bench with every kernel in its own grid. It is what turns 1.390 GB on card and an unmeasured throughput into measurements.

## 2026-09-06 (continued). Chantier 1: the attribution transposes, and `v_proj` is where the bits go

- **The quality roadmap is sanctioned** ([ROADMAP-QUALITY](ROADMAP-QUALITY.md), operator, 2026-09-06): 22 leads ordered by feasibility, gains in MMLU points only, the 2-bit landscape, and what the field does that we do not. It replaces `ROADMAP` section 2.3, which priced an arm at $7 on a `Planes14` base that is no longer the object. An adversarial pass over the 124-lead survey then corrected thirteen rows and collapsed four to zero: Q4a is zero structurally, since a block of 24 groups rotated coordinates; the rotation seed is refuted by our own journal, where the worst perplexity carries the best MMLU; Block-AP and PV-tuning both credit their gain to parameters `Tetra` does not have.
- **Chantier 1 is measured, with its prereg stamped first** ($0.60, 20 min, sha256 `866aaa26…`; [journal](mesures/q5-tetra-2026-09-06.txt), [deviations](../proofs/preregistration-q5-tetra-2026-09-06-ECARTS.md)). Four arms, one card, one process, one file, the same token fingerprint, and this time the per-question dump: T0 `Tetra` bare **53.49**, `v_proj` at f16 **58.10**, `v_proj` at int4 g128 **56.95**, the whole attention at int4 **58.48**. T0 replays the bench to the hundredth.
- **The attribution transposes.** Paired against T0: **+4.62 pp** [+2.33; +6.93], **+3.47 pp** [+1.42; +5.57], **+4.99 pp** [+2.18; +7.90], McNemar 5.0e-6, 8.6e-5 and 3.9e-8. All three resolved. The prereg's first line applies, since G4 = +3.47 clears +2.5 and its interval starts at +1.42. Survival from f16 to int4 is 0.751 stratified and 0.918 unweighted, against 0.804 and 0.944 under `Planes14`.
- **`v_proj` is twenty times the rate of anything else**: +3.47 pp for +0.0493 b/param is **70.4 points per b/param**, where the whole attention gives 10.1 and the increment between them 3.4. Restoring ten times more weights adds a third more gain. Sub-additivity, measured on `Tetra` rather than carried over.
- Against the served format: `Tetra` with `v_proj` in int4 reads **56.95 for 2.8138 b/param against 55.59 for 5.1619**, so quality at least equal for 55% of the memory (+1.36 pp, CI95 [−1.50; +4.22], which contains zero). The whole attention reads 58.48 for 3.2572, +2.89 pp with a stratified interval that grazes zero at −0.06.
- **Operator's decision, 2026-09-06: option B**, `v_proj` in int4 beside the `Tetra` matrices, code only, no re-encoding. Its reasons: the rate; the product margin, since B spends 6.4% of the 0.8502 b/weight before b_max where the whole attention spends 64.2% and forecloses Q6b, OWQ and Q4b; and the fact that B and C cost the same engineering, so B does not close C.
- Two corrections the evening forced, both in the deviations. **The tail f32 to f16 finances nothing**: the card has held it in f16 since 2026-08-09 (`TAIL_BYTES = 2`), so the published 2.7645 already includes it and the margin before b_max is 0.0747 b/weight larger than stated. And the control arms yielded for free what the 4B bench could not, having written no dump: `Planes14` against `Tetra` is **+2.59 pp, CI95 [−0.32; +5.58]**, which contains zero, with **650 of 2,280 questions discordant**. The format reshuffles answers rather than degrading uniformly, and the 4B deficit is not resolved. On the two-file paired SE of 1.50 pp measured here, the 8B's 3.91 pp is about 2.6 SE, not the 3.5 its journal states.
- The harness transfer to Metal was measured and then set aside: seven questions out of 2,280 flip against the card, McNemar p = 0.45, but an arm takes 50 minutes there against 4 on an L40S. The split that holds is **MMLU arms on the card, encodings on the Mac**, where the card would cost $7.
- Signed prediction: two right, two wrong. Gf and G4 land inside their brackets; the survival rate falls under its floor on the stratified reading and inside it on the unweighted one; Ga misses its floor by 0.01 point.
- Wave 2 closes at **$2.32 against a $2.00 cap**, a 16% overrun the operator arbitrated at launch. Project total $99.88.

## 2026-09-06 (continued). Tetra at the 8B: the memory holds, the quality turns

- **The 8B is encoded and measured** ($0.90, three arms, [journal](mesures/tetra-8b-2026-09-06.txt)). Encoded on the Mac in **4 h 41** (468 s/block, 36 blocks), all 6,945,767,424 weights read back bit for bit, sealed at 4,324,244,913 bytes against the published file's 4,324,243,889, the two weigh the same because the archive writes 48 bits per block either way. All of the difference is in VRAM, at the unfold.
- The campaign is three arms on one L40S, one harness, the same token fingerprints: f16 8.9899 / `Planes14` 10.9682 / **`Tetra` 11.0478** in perplexity, and 65.52 / **61.61** in MMLU micro. The published arm replays its 10.97 and its 65.52, so the harness is intact. The AWQ arm was dropped to keep the job inside the wave-2 cap.
- **The memory holds: 3.0672 b/param against 5.3220, ÷1.735.** It is a worse ratio than the 4B's ÷1.867 for an entirely accountable reason, `Tetra` does not touch the embedding, which is 9.67% of the 4B's parameters and **15.20% of the 8B's** (untied heads). In b/weight kernel the ratio improves instead, ÷2.235 → ÷2.270.
- **The quality does not: both axes turn against `Tetra` between the two sizes.** Perplexity goes from better by 4.64% to **worse by 0.73%**; excess log-likelihood from 0.854× the served format to **1.036×**; MMLU from −2.10 pp to **−3.91 pp**. And the MMLU gap is resolved this time and was not at the 4B: 3.91 pp is ~3.5 paired SE against ~1.5 for the 4B's 2.10. The exact paired interval needs the per-question dumps, which are in the bucket and cost $0 to pair.
- The perplexity read **during encoding** predicted the card's to the fourth decimal: ×1.2287 against ×1.2201 in-process on Metal in f32, ×1.2289 against ×1.2201 on the card in f16. The wrong sign was on screen five hours before the job was launched, and no one read it.
- Three causes are confounded in that gap and nothing separates them: the format, the encoder drift of 2026-08-26, and the fact that the published 8B was quantized **on a card**, where `calib.rs` accumulates AᵀA in f32 on the accelerator, while `Tetra` was encoded on Metal.
- **The bench ran without a prereg**, against hard rule 2, which is written without condition. It cannot be repaired, stamping now would attest to bytes written after the measurement, so its numbers are raw facts that **gate nothing**. Any decision they inform needs a fresh prereg before its own measurement. The 8B is outside the publication perimeter, which limits the damage without excusing it.
- Cost of the day: $1.69 over two benches. Wave 2 stands at **$1.72 of $2.00**, project total $99.28.

## 2026-09-07. Chantier 3: the paper's codebook is exonerated, and perplexity moves a fifth while MMLU does not move at all

- **`leech0c13`, the paper's own LLM configuration, encoded at the 4B for the first time** (3 h 55 on the Mac at $0, then $0.33 of L40S; prereg stamped before the first second, sha256 `9ecd46f8…`; [journal](mesures/leech0c13-2026-09-07.txt), [deviations](../proofs/preregistration-leech0c13-2026-09-06-ECARTS.md)). One variable against the `Tetra` of 2026-09-06: the codebook. Same model, same C4 corpus, same 131,072 tokens, same rotation, same device.
- **The rate is identical to the fourth decimal**, 2.0702 on both, 48 bits per block, 383 classes at no gain bit and shell 13. The prereg's section 4, which would have sent everything back to b/param had the rate moved, does not fire: the comparison is at constant bits, and that could not be known in advance.
- **Perplexity 19.6093 against `Tetra`'s 16.1569, MMLU 54.67 against 53.49.** The paper's codebook is **21.5% worse in perplexity** here and indistinguishable in MMLU. Paired: +1.19 pp against `Tetra`, CI95 [−1.54; +3.98], McNemar p = 0.79; −0.92 pp against the published `Planes14`, CI95 [−3.55; +1.78]. Both contain zero.
- **The fact of the day: our three arms (53.49, 54.67, 55.59) are indistinguishable in MMLU while their perplexities span 16.16 to 19.61.** The codebook choice moves a fifth of the perplexity and nothing measurable of the MMLU. It is the sharpest instance yet of the dissociation the file has been carrying since the 4B bench.
- **The prereg's third line applies: the codebook is exonerated.** The 5.1-point gap to the paper is not there. It is in what surrounds the codebook, a calibration volume about 95 times ours, the corpus, the rotation. The served-codebook debate closes, and calibration and fine-tuning take the head of [ROADMAP-QUALITY](ROADMAP-QUALITY.md). The reservation is written in the deviations: the interval's upper bound reaches 57.5, so the boundary with the second line sits inside it; what is robust is that 58.5 is outside, so the codebook does not carry the bulk of the gap.
- The signed prediction was **right**: 53 to 57 predicted with 55 at the centre, 54.67 measured. Its stated reason was the calibration volume, and the measurement agrees with it. Twelve signed predictions on this file, seven wrong, five right.
- **The in-process perplexity predicted the card's for the third time**: 19.6243 on Metal in f32 against 19.6093 on the card in f16, 0.08% apart, after the 8B's ×1.2287 against ×1.2289. The quality of an encoding in perplexity can be read for **$0 before a card is paid for**, which is now an instrument and not an anecdote.
- Cost: $0.33. Project total **$100.21**.

## 2026-09-13. L36's capture half, and the quality roadmap folded onto the served object

- **The capture pass exists** (`capture_model_hessians`, the `HessianSink` trait, `HBasis`,
  `Hessian::with_moments`, and the `hcapture` binary): one pass over the calibration set with the served
  weights, keeping the dense Hessian the encoder throws away. It quantizes nothing. The served loop is
  untouched, a capture needs no `GptqFactor`, so it gets its own loop, and one forward per block does
  both the accumulation and the next block's input because the weights do not move.
- `CalibCorpus`, `calib_chars`, `window_starts` and `ROTATION_SEED` moved out of `bin/smoke.rs` into
  `corpus.rs` and `calib.rs`. Not a tidy-up: a capture has to draw the *same* windows as the encoding it
  describes, and a copy would have drifted invisibly.
- Wiring check on the served 4B, 2 × 128 tokens, **$0, no card**: 144 emissions over 36 blocks, capture
  7.2 s, **reduce 126.3 s** (*measured*, [l36-capture-2026-09-13](mesures/l36-capture-2026-09-13.txt)).
  The cost is in the reduce (the f64 readback and the rotation), which is 17.5× the capture and does
  **not** scale with the calibration volume. The survey's "one Mac hour" rested on pass 1's 394.9 s,
  which is the capture phase alone; the real total at the served volume is *estimated* under 15 min.
- Ten tests, five mutants, all five dead. Two of them survived the first version of the tests, which
  checked length, sign and non-zero, every shape property and no arithmetic one. What kills them is
  `Σ‖x‖²/N = tr(H)` and Jensen's `‖E[x]‖² ≤ tr(H)`, both exact and both one line.
- **L36 decides eight rows, not the ten the survey claimed**: L06's cell asks for a paired re-encode and
  L15's for kurtosis on `f1recdump` blocks, and neither reads `H`. The consumer, `hstats`, is unwritten,
  so no row is decided yet.
- **[ROADMAP-QUALITY](ROADMAP-QUALITY.md) folded with the survey of 2026-09-12**, on the operator's
  instruction. Replaced, and recorded here with its date: its accounting header rested on bare `Tetra`
 , *"Base: `Tetra` at the 4B, 2.7645 b/param, MMLU 53.49, perplexity 16.1569. … The margin before the
  product triplet's b_max is 0.7679 b/param. … An MMLU arm costs $0 on the Mac"*. All four are now
  false or misleading: the base is the served mixed object at 2.8138 b/param and **56.37 on the full
  split**; the margin is **0.7956 kernel b/weight** (0.7679 b/param was bare Tetra's 0.8502 kernel
  b/weight converted at 0.903255, and mixing the two accountings is the subtraction to refuse); and an
  arm on the full split, the only plan that resolves anything, costs **$0.70**, not $0.
- Eleven cells of that table carry a dated correction in place, and eighteen rows are added. Row 6's
  gain moved from "0 to +2 on STEM *estimated*" to **+0.8 to +1.4**: the +2.98 pp of the ×32 arm is
  card-vs-card on a *naked* base, and the direct served-base reading is **+0.22**.
- **Still owed, and not done here**: `ROADMAP`:44 and :297 and `ETAT` §7 are stale; this file owes ten
  more entries from 2026-09-07 to 2026-09-12 covering **$42.14**; and five rows of `data/jobs.csv` cite
  three journals that exist in no commit of any ref, `volume-2026-09-07.txt` ($19.54),
  `q5-sur-v32-2026-09-07.txt` ($0.32), `m3-gptq-2026-08-30.txt` ($0.05). The first two carry the
  numbers row 6 now rests on, so writing them needs the bucket.

## 2026-09-14. Tetra gains-only Schur pilot

The local pilot completed in 10.77 seconds at zero remote cost (*measured*, [journal](mesures/tetra-schur-pilot-2026-09-14.txt)).
The 5–15 minute estimate was too high. Protocol and input manifest were timestamped before capture.
Schur lowers mean selection regret in both seeds, but only five scheduled decisions differ from Euclidean post-shape selection.
Three are terminal blocks. The two interior cases have opposite signs on the regularized rollout loss.
The signed prediction holds in seed 1 and understates the relative reduction in seed 2.
The former guide's unrun status is replaced by [the completed results](archive/tetra-schur-pilot-2026-09-14.md).
No served policy, kernel or quality claim changes.

## 2026-09-14. The Tetra gain rule against the Euclidean optimum

- **The served gain rule is not the Euclidean optimum, and the gap is measurable without a model.** A Tetra block decodes to `c_g · row_scale · u`; the served encoder picks the centroid nearest `‖x‖ / row_scale` (`quantizer.rs:777`), while the argmin of `‖x − a u‖²` over the same two levels is the one nearest `⟨x,u⟩ / row_scale`. Since `⟨x,u⟩ ≤ ‖x‖` the correction is one sided and can only move a block down. On 42,400 Gaussian blocks of the F1b source the two rules differ on **10.892 %**, 4,618 down and 0 up, mean `cos(x,u)` 0.960791 (*measured*, `llvq-bench/src/bin/gaindisagree.rs`, [journal](mesures/gain-desaccord-tetra-2026-09-14.txt)). Cost: 22 s of CPU, $0.
- **The same measurement on real compensated residues, and it needed no capture.** 2,016 shadow blocks of Qwen3-0.6B, two seeds, `q_proj` and `gate_proj` at layers 0, 13 and 27: disagreement **10.119 %**, 204 down and 0 up, mean `cos(x,u)` **0.961088** (*measured*, `ops/gain_disagree_real.py`, [journal](mesures/gain-desaccord-reel-2026-09-14.txt), [per-cell CSV](data/gain-desaccord-reel-2026-09-14.csv)). The synthetic run's own prediction is confirmed on both halves: the angular geometry is a codebook property and transfers to 0.03 %; the norm distribution does not, and compensation moves occupancy at the upper level from 45.901 % to 56.696 % and the median `‖x‖ / row_scale` from 0.9860 to 1.0120. Cost: 0.15 s of CPU, $0.
- **The centroid scalar reverses on real blocks, and one aggregate was hiding eight cells.** Shrinking the fitted pair by the mean cosine is worth −0.8818 % of squared error on Gaussian blocks and **+0.4705 %** on compensated ones, and applied with the rule it removes two thirds of the rule's gain (−1.9789 % against −0.4048 %). The rule itself holds: −1.2224 % synthetic against **−1.1311 %** real. Per cell the rule wins in **12 of 12**, from −0.6475 % to −1.5287 %, and the scalar loses in 8 of 12 (*measured*, same journal). The mechanism is the row above: the centroids are fitted on uncompensated weights, compensation then raises the norms the decision reads, so the served pair already sits low and shrinking it over-corrects. Stage 3 is closed, not deferred: an audit the same day swept the rescaling of the pair over [0.90, 1.15] and found the best at **1.0145**, *upward* as the norm distribution predicts, worth **−0.0258 %**, a fortieth of the rule; a full Lloyd-Max refit fitted and scored on disjoint seeds reads −0.11 % at best on `⟨x,u⟩` and +1.55 % at worst on `‖x‖`; and with each rule at its own best scalar the Euclidean rule still wins 12 cells of 12 by 0.55 to 0.87 points, so the gain is in the choice and not in the placement.
- **A plan reported absence four times from a session that could not see the working tree.** [plan-decision-gain-tetra](archive/plan-decision-gain-tetra-2026-09-14.md) §2 recorded `tetrapost`, the Schur code, the pilot directory and the 10.77 s figure as having no occurrence anywhere. All four existed, uncommitted, in the Documents working tree and in no other copy, not on `origin`, not in a stash. They are now `recherche/tetra-schur-2026-09-14` at `09e0f65`, and it is their dumps that answered stage 0 bis for 0.15 s instead of the budgeted day. The verdicts were true of the committed tree and false of the machine; a session that cannot see the tree can report failure to find, never absence.
- Nothing above is a quality claim. Retention is Shannon retention at a fixed rate, two transpositions from perplexity, and any MMLU arm on this lever reports noise until row A of [ROADMAP-QUALITY](ROADMAP-QUALITY.md) lands. The anchor stands: 21.5 % of perplexity between `leech0c13` and Tetra bought 1.18 pp of MMLU, CI95 [−1.54; +3.98].
## 2026-09-15. The Euclidean gain rule, end to end: dead

- **A local gain of 1.13 % bought a loss of 4.026 %, and the lead is closed.** Two encodings of Qwen3-0.6B, identical in every respect but the word `tetra` against `tetrapost`: perplexity **41.8875** against **43.5739**, degradation ×2.148 against ×2.234, effective rate **2.1656 b/weight on both** (*measured*, 31 min of Mac, $0, [journal](mesures/tetrapost-ppl-0.6b-2026-09-15.txt)). The f32 baseline reads 19.5038 on both arms, which is the M1 journal's own value of 2026-09-02, so this is that protocol at the digit.
- **The prediction was signed, stamped and refuted.** The prereg predicted B below A by 0.2 % to 1.2 %; B came out above by 4.026 %, wrong sign and three times outside its own interval ([prereg](../proofs/preregistration-tetrapost-ppl-0.6b-2026-09-15.md), [deviations](../proofs/preregistration-tetrapost-ppl-0.6b-2026-09-15-ECARTS.md)). Its own clause, beyond 3 %, doubt the measurement before the hypothesis, fired and was discharged by the four controls: same rate, same baseline, same 440,401,920 weights, one word of configuration apart. The decision rule stopped the queue after the first pair, so seeds 1 and 2 were never run: 31 min spent of the 3 h authorised.
- **The mechanism is measured, not inferred.** The Euclidean rule moves every disagreed block *down* a level, 204 of 204 on real blocks, 4,618 of 4,618 on synthetic ones, because `⟨x,u⟩ = ‖x‖·cos θ ≤ ‖x‖`. Mean amplitude placed over block norm: **0.99336** under the served rule, **0.97073** under the Euclidean one, a systematic **2.9 % shrink** (*measured*, same pilot dumps). It buys squared error by trading an unbiased error for a smaller **biased** one. A centred error partly cancels down 28 layers; a one-sided shrink of every block accumulates.
- **The plan's §9 is the claim that failed, and it is the transferable lesson.** It argued this lead was unlike the three precedents (design C, `group_scales`, gptq2) because rule C leaves the compensation loop, the row scale, the direction and the format untouched. It checked that the procedure around the decision was intact and never checked what the decision does in aggregate. Changing which of two admissible codes is written changes the distribution of the amplitude written, the same class of defect as `group_scales`. **A selection rule is not local just because each of its decisions is.** The repository now holds four cases, not three.
- Kept from the dead lead: `ops/gain_disagree_real.py` and its audit, which answered a one-day question in 0.15 s from dumps already on disk; and the fact, still standing, that the served rule and the Euclidean optimum disagree on 10.119 % of real blocks. What is refuted is that the disagreement should be resolved the Euclidean way.

## 2026-09-17 and 18. The state reconciled, and two accounting errors found in it

A container reading a 2026-09-11 HEAD concluded that the experiments of 09-16 and 09-17 were
missing. They were on the Mac, in 57 commits ahead of `main`. The state is written once, for
`main..HEAD`, in [`etat-reconcilie-2026-09-17.md`](archive/etat-reconcilie-2026-09-17.md), with each
result tied to its code, parameters and data. Operator decision: no merge into `main`, and the
38 leads of `claude/fine-tuning-cost-mmlu-gain-kic9ua` stay out of scope.

Four distinct quantities are called rho in the repository. The negative conditional rollout of
09-16 ran at rho_H = 1.0, inherited from the Schur pilot bundles, and `tetraalt.rs` has no rho
of its own. Under rho_H, selection and compensation come from one `GptqFactor` and are
inseparable; row scales never move with it.

Every projection type fits under b_max in int4 natively. The allocation journal of 09-16 set
whole-model b/param against a b_max in kernel b/weight, and only the o_proj row was ever
redone. Redone for all six: `down_proj` reads 2.7226 kernel b/weight, not the 3.2286 that put
it out of budget, and v + o + down reads 2.9408 (*computed*). The condition is a native int4
kernel for the shape, which has run on `v_proj`'s alone.

The served object's recorded sha256 was the sha256 of its encoding prereg, propagated from a
journal's `Prereg` line into the f1e census prereg and then into `configs/README.md`. The
object's own digest is `ae31087a...2263` (*measured*, two bucket downloads,
[references-comptabilite-2026-09-18](mesures/references-comptabilite-2026-09-18.txt)). The byte
count was right, which is why every size check passed. And the served b/param read off the file
is 2.7475 at the f16 tail the card has held since 2026-08-09, against the 2.8138 *computed*
that `configs/README` carried.

## 2026-09-19 and 20. The row scales, the tile, and a base that passes the paper's non-fine-tuned line

Training one scale per weight row is worth **+4.30 pp of MMLU for zero bits** on bare Tetra (*measured*, 1.71 h on
l40sx1, about $3.90, [tetranu-rowscales](mesures/tetranu-rowscales-2026-09-19.txt)). The scales already exist in the
file; only their values change, so the rate does not move.

Replayed on a DCLM-calibrated base the next day, it gives **57.95 to 61.11 at 2.7475 b/param** (*measured*, about
$4.00, [dclm-rowscales](mesures/dclm-rowscales-2026-09-20.txt)). That passes the source paper's own row for a 4B
without fine-tuning. Perplexity goes to 1.0074 times f16, and MMLU stays nine points under the dense model: the largest
of the six dissociations recorded here.

The tile is measured, not guessed. At the served settings, 64 beats 128 by **+16.1%** on sm_89 and 1.5% for `Planes14`
(*measured*, about $0.25, [tuile-l40s](mesures/tuile-l40s-2026-09-20.txt)). It costs zero bits and the output is
bit-identical: the tile steals L1 from the decoder table. `LLVQ_TILE_BLOCKS` unset now reads the measured row for the
card. Every figure published before this date was measured at tile 128.

## 2026-09-21 and 22. The 8B chain: the kernel serves it, the census places it

The Tetra kernel serves the 8B at **90.6 tok/s in 3.15 GB, 256 tokens identical to the dense arm** (*measured*,
[served-8b](mesures/served-8b-2026-09-21.txt)). The census puts the paper-2 base at **64.87**, ten points under f16's
75.05, with AWQ at 73.79 on the same 14,042 questions (*measured*, [census-8b](mesures/census-8b-2026-09-21.txt)).
Training the row scales lifts it to **68.16**, +3.29 pp [+2.78, +3.80] (*measured*,
[dclm-8b-rowscales](mesures/dclm-8b-rowscales-2026-09-21.txt)). The chain cost $10.19.

The kernel bench of that run is formed on unequal work: 216 matrices for Tetra against 252 for the other arms, because
the int4 `v_proj` are counted and not timed. Any × read off it compares different work, and paper 2 carries it as a
limitation rather than a ratio.

## 2026-09-22. Spherical GPTQ on Tetra, refuted

The paper's Algorithm 3 keeps every block on its exact norm inside the loop, so the residual it propagates carries
angular error only, and Table 9 credits that with +1.9 pp of MMLU. Under a coded gain the published path cannot do it.
`GptqConfig::spherical_feedback` does, changing nothing else.

End to end on the 0.6B, two arms, one file, $0: perplexity **41.8875 against 55.2478, so the angular feedback is worse
by 31.89%** (*measured*, [sph-ppl](mesures/sph-ppl-0.6b-2026-09-22.txt)). The preregistration predicted it better by 1
to 4%. It is refuted on the first pair, in the opposite direction, at eight times the width of its own interval. Fifth
case of a locally better reading composing worse once the gain is coded.

## 2026-09-22 and 23. The 14B chain, on a card

Encoded in two segments on `rtx-pro-6000`: R = ×1.1709, sealed 6,563,782,117 bytes, **2.7371 b/param**, the four
figures of the prereg exact to the ten-thousandth (*measured*, [encode-14b](mesures/encode-14b-2026-09-22.txt)). The
export of 29.5 GB was lost to a silent mount truncation and redone in 30 slices for $0.32.

The base reads **72.53** against f16's 78.88 and AWQ's 78.12 (*measured*,
[census-14b-base](mesures/census-14b-base-2026-09-22.txt), [census-14b-ref](mesures/census-14b-ref-2026-09-22.txt)).
The trained scales give **74.20, +1.67 pp [+1.27, +2.08]**, half of what the 4B and 8B gained, and perplexity 9.3447 to
8.4622 against AWQ's 8.2858 (*measured*, [dclm-14b-rowscales](mesures/dclm-14b-rowscales-2026-09-22.txt)). The fold ran
**against** the training's own `improved` gauge, on an operator go: the gauge compares the last third to the first and
is confounded by a streaming corpus (deviation É3). The chain cost $32.17 of a $52 cap.

## 2026-09-23 and 24. The three files are sealed

The int4 embedding pays for two int4 projections. At 4B, storing the tied table int4 g64 frees the bytes for `o_proj`
and `down_proj`, and MMLU goes **61.11 to 63.37 at 2.7320 b/param**, +2.26 pp [+1.66, +2.87] (*measured*,
[embed-q4-swap](mesures/embed-q4-swap-2026-09-23.txt)). The same move at 14B gives **75.66**, and at 8B **69.58**
(*measured*, [sealed-8b-14b](mesures/sealed-8b-14b-2026-09-23.txt)).

At 8B the int4 budget was spent two ways at equal bits. `down_proj` of layers 10 to 26 alone scores 69.58 at 2.6953,
beating `o_proj` plus `down_proj` by **0.77 pp [0.26, 1.28]** with fewer bytes, which is the opposite of our signed
prediction (*measured*, [sealed-8b-27](mesures/sealed-8b-27-2026-09-24.txt)). The 4B and 14B still spend it the old
way, untested.

A second training on the sealed 4B, this one also learning the 73 RMSNorm vectors, lowers the KL loss and moves MMLU by
nothing: **63.40 against 63.37**, +0.02 pp [−0.31, +0.36], p = 0.93 (*measured*,
[sealed-rownorms](mesures/sealed-rownorms-2026-09-23.txt)). The file is kept without it.

## 2026-09-25 and 26. The paper table, and paper 2 rewritten

The served files decode at **113.8, 95.0 and 57.2 tok/s with 1.38, 2.76 and 5.04 GB of weights** (*measured*,
[paper-table](mesures/paper-table-2026-09-25.txt)). Two defects in our engine surfaced first, before any number: a
group with no rotation sent all prompt rows to the matrix-vector product at once, which is wrong for the int4 kernel,
and that kernel was capped at the default 48 KiB of shared memory, which a 14B `down_proj` exceeds. Both are fixed and
tested. Neither touches an MMLU, which is scored on the dense reconstruction.

Paper 2 is rewritten as a single-column arXiv paper on the three sealed files: 17 pages, `RELEASE=1 make check` green,
no pending cell. Three external reviews converged on one rule, applied throughout: reduce a claim to what its
experiment establishes at its first occurrence, rather than neutralize it later. Withdrawn and not to be reintroduced:
a decoder ratio formed on unequal support, "two thirds of the time is launch geometry", "limited by decoding", any
speed ratio between two engines, and any cross-harness ranking.

## 2026-09-26. The repository catches up with its paper

Three journals the paper's provenance table names had never been committed: the 14B reference census, the 14B base
census and the 14B row-scale training. They were on the Mac, untracked. Every journal, dump and preregistration the
paper cites is now in the repository, and the four stamps attest the bytes committed.

The living documents were three weeks behind the object. `ETAT.md` still gave `Planes14` as the served configuration at
55.59 MMLU, `README.md` and `CLAUDE.md` still pointed at the earlier file. They are rewritten on the sealed Tetra
objects and brought back to the lengths `STYLE.md` sets: ETAT 566 lines to 134, ROADMAP 407 to 117, README 187 to 126,
CLAUDE 161 to 146. The entries of this file between 2026-09-05 and 09-17 were out of chronological order and are
reordered.

## 2026-09-26 and 27. A reasoning test: GSM8K through the served kernel

The three sealed files score **82.49, 88.63 and 92.04 on GSM8K** through the served kernel, against 92.12, 93.25 and
95.30 for FP16 and 89.01, 92.95 and 95.38 for AWQ in vLLM (*measured*,
[gsm8k-wave1](mesures/gsm8k-wave1-2026-09-26.txt), [gsm8k-wave2](mesures/gsm8k-wave2-2026-09-26.txt)). It is the
first quality score read on the served object over a whole test split. The paired gap to FP16 is 9.63 points at 4B,
1.4 times its MMLU gap. At 8B and 14B it is 4.62 and 3.26, and does not differ from the MMLU gap.

A new harness generates and grades through the kernel (`bin/gsm8k`), vLLM generates the references on the same prompt
ids (`ops/gsm8k_vllm.py`), and one grader scores every arm (`bin/gsm8kpair`). A free Metal pilot priced the campaign
first ([gsm8k-pilot](mesures/gsm8k-pilot-2026-09-26.txt)). The engine gate held: FP16 reads 91.51 in our dense path
against 92.12 in vLLM. The campaign cost 14.26 $.

## 2026-09-28. One size up at the same memory

The review of paper 2 found no comparison at equal memory. The committed dumps answer it for free: one size up, our
file beats AWQ on MMLU and ties it on GSM8K (*measured*, [sizeup](mesures/sizeup-2026-09-28.txt)). Our 8B scores 1.44
points above AWQ's 4B and our 14B 1.87 above AWQ's 8B; on GSM8K they tie at −0.38 and −0.91. The reasoning loss takes
back the larger model. Our 14B in 5.04 GB scores 5.52 MMLU points above FP16's 4B in 8.04 GB and ties it on GSM8K.
The 4B and 8B share one KV cache, so the step from 4B to 8B costs weights only. Paper 2 carries it as section 6.4 and
Table 5, marked as not preregistered.

## 2026-09-28. Paper 2, every number audited

1,132 numbers of paper 2 were checked against their CSVs, journals and code, by eight auditors,
eight skeptics and a blind sample; about a hundred corrections went in, each checked by hand
against its source (*computed*, [paper2-audit](mesures/paper2-audit-2026-09-28.txt)). The main
ones: the calibration spread of three draws is a range of 5.83 points, not "2.92", which is its
standard deviation; the 4B MMLU was scored one step before sealing; the cited LLVQ scores use the
0-gain-bit codebook; the cost of the Tetra split was measured, not only estimated. Four results
are new: the shell bound is m <= 26, found by an exhaustive pass that replaces a sweep which was
not a bound; Tetra against Planes14 on the 2,280 sample is +2.11 [-0.95, +5.17], not separated;
the size-up budgets move by one hundredth once summed from unrounded bytes; and the decode costs
about 150 instructions a block, not 141. The paper now runs to 19 pages. Two fresh readers
then checked every changed line and caught three more contradictions, among them the tail
width in the caption of Table 2.
The same day, the 24 cited works and the 67 provenance rows were checked against their
publications and files: 55 problems, one dead URL (the CUDA guide), and a dozen attributions
narrowed to what the cited work says, the chief one that the cited QTIP scores and the timed QTIP
kernel are two different codes. 20 pages.

## 2026-09-28. Paper 2, from 20 pages to 14

The same paper in 14 pages, with no number changed and three intervals added (FP16 minus AWQ, from paper2-gaps.csv).
Four readers proposed 212 cuts and four skeptics tried to refute each; five editors applied the kept ones and five
verifiers checked every diff. The provenance table (2.3 pages) moved to `paper2/PROVENANCE.md` and the commands to
`paper2/README.md`. The tile figure, the scale figure and panel (a) of Figure 2 went, their values kept in the text.
The defects paragraph, an erratum about a draft never distributed, the second training and four limitations that
restated a caveat of the body went too. Each caveat of the audit is stated once. The intro and the conclusion now name
section 6.4, marked as chosen after measuring. Appendix A and its QTIP row, the full predictions table and Figure 4
stay.

## 2026-09-28. Paper 2, four review points and a proofreading pass

A reviewer's four points, all taken, and the paper grows back to 15 pages. "Tie" becomes "not separated", defined once:
no equivalence test was run, and the GSM8K intervals of the one-size-up pairs reach down to -2.14 and -2.35 points. The
14B divergence at token 78 is reported as an observation: it moves to token 137 with f16 tables, the prefill gate shows
logit differences up to 1.35 between two paths of the kernel, and the margin at token 78 was never measured. The paper
now says the int4 choices were made on MMLU test questions, with no held-out split; the one measured case, `o_proj`,
reads +3.12 on the questions that chose it and +1.55 on the other 11,762 (*measured*,
[oproj-int4-full](mesures/oproj-int4-full-2026-09-17.txt)). GSM8K chose nothing. With AWQ's tables in 4 bits, our files
would need 64 to 65 % of its bits per parameter, not about half (*computed*, [sizeup](mesures/sizeup-2026-09-28.txt)).
A proofreading pass fixed six slips, among them the running header, whose small-caps ligature extracted as "qantized".

## 2026-09-28. Stage 0 of the transformers plan: the sealed 4B as safetensors

The served 4B has a faithful safetensors image, 1602 fields rebuilt bit for bit by an independent Python reader
(*measured*, [hf-safetensors-4b](mesures/hf-safetensors-4b-2026-09-28.txt)). The directory is 1.422 GB against the
sealed file's 1.418, so the object stays compressed: `bin/export` writes 8 GB of f16 and is an interchange artifact,
this one is a distribution candidate. Three format decisions were taken before the code and recorded in the prereg: our
own tensor naming, the disk's bytes as the code payload rather than the served `tetra48` layout, and the rotation
carried as its two tables rather than its seed. `code_stream` and `code_widths` are now public in `llvq-artifact`, and
`write_matrix_raw` calls them, so a packer cannot disagree with the format about a width or a byte order. The gate was
read three ways, one mutant each, all caught; the f16 fields caught a real defect first, the reader casting f16 to u16
instead of reading its bytes, which hashed a scale of 0.001 as a zero. One signed prediction missed: the
`quantization_config` block is 131 KB against a predicted 20 to 60, from pretty printing. Nothing loads the directory
through `from_pretrained` yet, which is stage 1.

## 2026-09-30. Stage 1: transformers reads the packed 4B and answers

`from_pretrained` on the packed directory gives the weights the artifact defines and the tokens our
engine gives: 253 of 253 dequantized weight digests identical to `decode_matrix`, and 256 of 256
greedy ids identical to `bin/run` over four prompts, f32 on the CPU on both sides (*measured*,
[hf-quantizer-4b](mesures/hf-quantizer-4b-2026-09-30.txt)). No missing key and no unexpected key.
The Leech decode is ported to numpy over the tables `bin/tetratables` dumps from
`llvq_search::tetra`, 19 KB shipped with the package rather than with every model, because the map
belongs to the codebook. Bit-exactness was predicted and held on the first run, including the one
place named as the risk: the `k` by `k` mix, accumulated term by term rather than through a matrix
product, which a library would be free to reassociate. Two facts about `transformers` 5.17 shaped
the code: `create_quantized_param` no longer exists, so a quantizer replaces its modules and the
ordinary loader fills them, and that loader keeps the dtype of an existing buffer, which is what
lets the f64 row scales survive a load at f16. Replacing the `nn.Embedding` broke the tie to
`lm_head`, since `tie_weights` runs before the quantizer's post-load hook, so the quantized
embedding goes through the conversion pipeline instead, three keys to one parameter. The Python
side lives in `llvq-hf/` and is extracted at stage 5.

## 2026-09-30. Stage 2: the served shader decodes the 4B under PyTorch

`torch.ops.llvq.tetra_decode` dispatches `tetra48_probe`, the shader's own decoder entry point, and
every one of the 118,665,216 blocks of the packed 4B decodes to the point the numpy path gives:
2,847,965,184 coordinates, exact, in 38.4 s (*measured*,
[hf-metal-decode-4b](mesures/hf-metal-decode-4b-2026-09-30.txt)). No MSL was written for it. The two
sides of that gate were already pinned to `llvq_search::tetra` before the stage began, one by
`llvq-metal/tests/tetra48_matches_rust.rs` and the other by stage 1's control 2, so the gate closes a
triangle. The plan's gate for this stage was unachievable and was narrowed rather than widened:
stage 1's dequantization is an f64 chain and Metal has no f64. The fused matvec moved to a stage 2
bis, blocked on `upload_int4` staging `d_in · 4` bytes against a 32 KB threadgroup limit, which
refuses all three sealed files on the Metal fused path. One defect was found and fixed on the first
dispatch: the transcode from the disk stream to the served layout is not a byte reversal, because the
gain bit moves from bit 0 of the disk value to bit 47 of the Tetra word.

## 2026-09-30. M1: the 4B answers with its Tetra weights never materialized

`tv_tetra48_metal`, the served shader, runs in a `transformers` forward pass: 256 greedy ids of 256
against `bin/run`, the 168 Tetra projections resident in 0.749 GB against 11.4 dense, 5.436 GB
allocated on the device against 16.1 computed for the dense arm, and the load down from 153.7 s to
6.5 because the fused path never dequantizes (*measured*,
[hf-metal-m1-4b](mesures/hf-metal-m1-4b-2026-09-30.txt)). What remains dense is the 21 % of int4
records, whose Metal kernel is blocked by a 32 KB staging limit, and the embedding. The rotation runs
in torch f32 rather than through `rot_apply_metal`, at 0.15 ms of the 0.25 a matvec costs.

The lot's real finding is about the method. Dropping the tail costs 8.79 % of the output on `k_proj`
and 3.53 % on `down_proj`, per row, and the token gate misses it over eight ids on all four prompts
and over all 64 on two of them. Stages 0 to 2 and M1 all gated on token identity; the per-row check
finds that same defect in 20 seconds with a 3,400-fold margin. Whether it becomes a gate is an open
decision.

## 2026-09-30. M2 and stage 4: no projection dense on Metal, the served CUDA kernel from torch

The 4B holds no projection dense on Metal any more. `tv_q4_metal_tiled` sits beside the served `tv_q4_metal` and gives
the same f32 values at tiles 256, 2,048 and 8,192 wherever the served kernel runs. The model gives the same 256 ids of
256, and the device allocation falls from 5.436 GB to 2.750, with all 252 projections resident in 1.158 GB
(*measured*, [hf-metal-m2-4b](mesures/hf-metal-m2-4b-2026-09-30.txt)). The four signed predictions held. One control
nearly went void: a mutation located by the kernel's name landed on both kernels, so the reference moved with the
subject. `checkq4guards.py` now closes the region at the next entry point.

On CUDA, `nvcc` compiles the served `llvq_tetra48.cuh` outside NVRTC and keeps its arithmetic: per-row 3.74e-04,
3.41e-04 and 3.53e-04 relative on three shapes against a 1e-2 bar, identical on two L4 instances, and 256 greedy ids
of 256 against `bin/run` (*measured*, [hf-cuda-4b](mesures/hf-cuda-4b-2026-09-30.txt)). The served header gained one
qualifier, `__device__` on `TETRA48_ORDER`, which changes no arithmetic. 168 of 252 projections are fused there,
because no torch op binds the int4 kernel. Seven launches cost $0.30, and two of the three build errors they found
are now held by tests on the Mac.

## 2026-10-01 to 10-02. The 4B published, and the package named

The out-of-tree route is complete and verifiable by a stranger. Two public repositories,
[Qwen3-4B-LLVQ-Tetra](https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-Tetra) for the safetensors that
stay compressed and [Qwen3-4B-LLVQ-Tetra-sealed](https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-Tetra-sealed)
for the single file the Rust engine reads. **The Hub computed the sealed file's SHA-256 itself and it
is paper 2's**, `886391a8c03f66dc`: until 2026-10-02 that digest named a file on one laptop, so nobody
outside could check it. The published repository loads unauthenticated from an empty cache, 252
records replaced, no key missing or unexpected (*measured*,
[hf-cleanroom-4b](mesures/hf-cleanroom-4b-2026-10-01.txt)).

**Three defects found by asking what a stranger gets, not by reading the code.** `import llvqhf`
registered nothing, and failed soft: `transformers` warned "Unknown quantization type, got llvq ...
we will skip the quantization", loaded the file as dense and raised fifty lines later about a
corrupted checkpoint. Four stages of measurement had passed over it, because every script of the
package imported `.quantizer` by hand. `hfpack` writes a 64 byte `tokenizer_config.json`, a stub,
where the real one is 9,732 bytes and carries the chat template, so the repository as packed would
have shipped a tokenizer where `apply_chat_template` fails. And the wheel carried no CUDA source at
all: `_extension_cuda` looked for the repository two directories up, which under `site-packages` is
nonsense, so a pip user had no CUDA arm.

**A loadable fixture, and the fixture is the lesson.** No test in either language had ever called
`from_pretrained`; `fixtures/tiny` describes nothing on purpose and the real object is 1.4 GB.
`fixtures/mini` is a coherent one-layer Qwen3 of 148 KB whose every dimension is forced: 136 = 17 x 8
so the rotation's odd part is real where a power of two gives the trivial 1 by 1, two different
non-empty tails at 16 and 8, int4 only where `d_in` is a whole number of groups. The first version of
its test left `import llvqhf` out and, run alone, **passed its forward pass on random numbers** with
three tests of five green. A forward pass that runs proves nothing; the gate is `missing_keys` and
`unexpected_keys`.

**The package is `llvq-tetra` from 2026-10-02**, directory and module and distribution name, against
`llvq-hf` and `llvqhf` before. Preregistrations, journals and archived documents keep the old paths:
they are dated records and rule 2 forbids editing a stamped file. The six CUDA sources now ship
beside the two Metal shaders, in the repository's two-level shape so that `tv_tetra48_h.cu`'s own
relative includes resolve, and `the_shipped_cuda_closure_is_complete` recomputes the include closure
from the bytes rather than trusting the hand written list. Stage 4 had passed on a card for $0.30
over seven launches, five of which bought three build errors and one a queue of 3 h 47.

## 2026-10-03 and 04. A published model that loads as random, and a conversation

A caller that imports `transformers` alone gets the published 4B as a randomly initialized model, with no exception:
254 missing keys reinitialized at Qwen3's own std of 0.0200, 1,119 unexpected, and logits in an ordinary range
(*measured*, [hf-tripwire](mesures/hf-tripwire-2026-10-03.txt)). `transformers` registers quantization methods by
import, never by entry point. The next step planned was a quality figure through `lm_eval`, which does not import our
package, so it would have scored noise. `auto_map` cannot close the trap while `model_type` is `qwen3`. The operator
decided on 2026-10-03 to add it and keep the type: `trust_remote_code=True` now loads the model, verified from the Hub
with an empty cache, and a caller who passes nothing still gets the random model, as the card says.

`llvq_tetra.chat` talks to the 4B. On two prompts beside the FP16 checkpoint, greedy, FP16 answered visibly better
(an impression, [hf-chat-4b](mesures/hf-chat-4b-2026-10-04.txt)). The dense arm loads in 151.6 s against 4.5 for
FP16, because it rebuilds 252 matrices before the first token; the fused arm loads in 7.7 s and generates more slowly.
The upstream issue for `transformers` was drafted on 2026-10-04 and not posted
([`upstream/transformers-llvq-tetra/`](upstream/transformers-llvq-tetra/)).

## 2026-10-04 to 10-06. Review campaign, paper 2 version 2, release v0.0.2

An automated review of paper 2 contested no number. Answering it in full was priced at about $285. The operator
funded $9, and the campaign spent $4.57 (*measured*, `docs/data/jobs.csv`).

The three sealed files got a perplexity, 12.58, 9.71 and 8.44, all read on the L40S. The 4B FP16 and AWQ references
had been labelled as Apple-GPU readings; both were L40S readings of 2026-08-06, so no ratio crosses devices. The last
build step raises perplexity at 4B and 8B while it gains MMLU. On a quantized 8B file, Metal and the L40S agree to
0.01 % (*measured*, [ppl-bases-carte](mesures/ppl-bases-carte-2026-10-05.txt)).

The kernel bench moved from 216 to 252 matrices for `Tetra`, on the bare encode. Its time went from 3.424 to 3.734 ms,
its rate from 2.148 to 2.150 b/weight, and its tile gain from 128 to 64 on sm_89 from +16.1 % to +17.6 % (`Planes14`:
1.5 % to 1.1 %). A mixed int4 arm was written for the purpose, then dropped on the operator's instruction.

On an A100, `Tetra` beats our FP16 kernel 1.11× but not cuBLAS 1.14×, and the served files decode below FP16 in vLLM
(*measured*, [a100](mesures/a100-2026-10-06.txt)). The prediction that `Tetra` would lose to FP16 there was wrong
against our own FP16 kernel.

The 8B and 14B files went public on the Hub beside the 4B. The 4B card had said the two were deleted in a disk
cleanup, but they had sat in the job bucket all along. A first publish of the sm80 Space uploaded a local virtualenv,
19,618 files, and the Hub's secret scanner flagged numpy's own test names in it. The folder was removed, and `publish`
now refuses a perimeter over 1,000 files.

A second number audit of the paper found seven errors, none in a table
([paper2-audit](mesures/paper2-audit-2026-10-06.txt)), before a redundancy pass and a style pass. Merged into `main`
as `344499d` and released as `v0.0.2`, with the paper as a PDF.

## 2026-10-07. The transformers route comes to `main`

The operator decided to merge `hf-safetensors` into `main`, and `llvq-tetra/` stays a package of this repository. That
reverses the decision of 2026-09-30 to keep the branch off `main` and extract the package. Three facts of 2026-10-02
made extraction a source of drift: the wheel ships a copy of the served kernels made by `bin/tetratables`, the `mini`
fixture is written by a Rust test, and `hfpack`, which makes the published object, cannot leave the workspace. Only the
glue would go to `transformers` in tree, the way `aqlm`, `vptq`, `spqr` and `higgs` keep their own packages.

The registry gains the six stage 4 rows `main` lacked, $0.30: $246.75 over 227 priced jobs. The safetensors card moves
to `docs/hub/qwen3-4b-safetensors.md`, byte for byte the Hub's README. The upstream issue and its PR go out together,
once PyPI holds 0.1.0.

Five defects of the package were fixed before the first upload, because a version on PyPI cannot be changed. The
import registered "llvq" unconditionally while `transformers` raises on a name it already holds, so 0.1.0 would have
failed to import on the first `transformers` that carries the method in tree. The import guard swallowed any
`ImportError`, so a `transformers` too old for the package, one without `transformers.core_model_loading`, left the
method unregistered and `from_pretrained` loading a random model in silence. `torch` and `transformers` were an optional
extra, while the card's install line is `pip install llvq-tetra` alone. And the floors, `torch>=2.5` and
`transformers>=5.0`, admitted torch 2.5.1, which breaks against transformers 5.x; they are now the versions measured,
2.14 and 5.17. Two tests were added, 48 pass, and three mutants, one per decision of the guard, were each caught by
exactly one test. The fifth came from `twine check --strict` on the built wheel: two project URL labels held a comma,
which core metadata reads as the separator between label and URL.

The full suite ran twice. The first pass stopped at the first failing binary, so the second ran without fail-fast:
180 suites, 1,036 tests passed and 6 failed (*measured*). Three read the August archive `~/qwen3-4b-llvq.bin`, absent
from this machine: `real_rows_match_the_sealed_q8_file`, `transcode_of_the_sealed_model_matches_planes14` and
`rotation_keys_partition_the_sites`. The operator deleted them. Three CUDA rotation tests raced: four tests compiled
the same host driver into one fixed path while another one ran it. The driver is now built once per test binary, and
`golay70_decoder_matches_rust.rs`, which had the same pattern with two callers, got the same fix. The fix passed five
runs at 16 threads. The old behaviour also passed five runs on an idle machine, so the race cannot be forced here.

`llvq-tetra` 0.1.0 went to TestPyPI, the token read from a local `.env` that git ignores. TestPyPI serves the bytes
that were built, and installed from it into an empty environment, with nothing compiled, the package loads the
published 4B and gives the 256 ids of `bin/run` (*measured*, [hf-testpypi](mesures/hf-testpypi-4b-2026-10-07.txt)).
The check found a version the prereg had not seen: transformers 5.19.0, out the evening before, is what
`pip install llvq-tetra` resolves. It ran as a second arm, written beside the prereg before any load, and gave the
same 256 ids, so 0.1.0 needs no upper bound.

On the operator's go, the same bytes went to PyPI. PyPI serves the digests TestPyPI served, and
`pip install llvq-tetra` in an empty environment pulls torch 2.14.1 and transformers 5.19.0 and loads the fixture with
no missing or unexpected key (*measured*, [hf-pypi](mesures/hf-pypi-0.1.0-2026-10-07.txt)). The install line of
both Hub cards, and of the shim's `ImportError`, had promised that package since 2026-10-02: it is true now.

## 2026-10-30 to 11-02. The decode step trimmed, attention ungrouped from `repeat_kv`, and a field to serve it

Four jobs on the L40S, one of them a failed first attempt, $1.55, all on the sealed 4B and its config. They answer the two cheapest questions a throughput
audit of 2026-10-29 raised: what the served step wastes on work it does not need, and what the KV cache costs at the
context the product promises.

**The trims.** Three removals, each bit for bit, give the 4B 122.4 tok/s [122.3-122.4] against 114.2 [113.6-115.2],
the path every published figure ran: the causal mask is not rebuilt when it is all zero in decode, `tv_q4_h` reads f16
where 84 conversions to f32 preceded it, and cudarc's event tracking is off on a stream that is alone (*measured*,
[trims-contexts](mesures/trims-contexts-2026-10-30.txt)). Singly they read 1.0283, 1.0230 and 1.0115; together 1.0716,
against 1.064 for their product. The same job measured the served decode at four prompt lengths for the first time:
122.5, 82.7, 61.4 and 33.2 tok/s at 5, 2,048, 4,096 and 8,192, with the first token at 0.03, 8.15, 16.91 and 34.86 s.
At 8,192 the cache takes 72.9 % of a decode token, which is what designated it as the next lever.

**Attention over the cached heads.** The shipped path concatenates the history every step and then copies each key and
value head to its four query heads, twice: `repeat_kv` in candle 0.9.2 is a `cat` of four copies. Two forms that copy
nothing were measured against it on a `Grow(256)` store, which grows by whole chunks and is written in place. The
grouped form, which stacks the query heads of one key head as rows of one matmul, reads 134.4 tok/s at a 5-token prompt
and 77.5 at 8,192, against 121.9 and 33.1, and returns the same 256 tokens at all four lengths (*measured*,
[gqa-ab](mesures/gqa-ab-2026-11-01.txt)). The shared form, which keeps the expanded matmul's shape, is slower
everywhere and was dropped.

**Neither unexpanded form is bit for bit on the card, and that changed a gate.** The shared form was bit for bit on the
Mac's CPU and is not on the L40S: cuBLAS rounds a batch of 8 key heads differently from 32 expanded ones, up to
6.6e-2 on a logit. Per-head shapes do not fix it. Both forms therefore fell under decision T8 of the audit, and the
operator took it as "tolerance plus control": a bounded tolerance replaces identity at the bit, provided a quality
control passes first. The control ran on WikiText-2 through the served kernel, 12 windows of 4,096, paired window by
window: +0.0000 % by prefill, bit for bit, and +0.0095 % [-0.0048 ; +0.0238] by decode, both inside the 0.12 % band of
[`ETAT.md`](ETAT.md) §4, with 99.788 % top-1 agreement over 49,140 steps (*measured*,
[gqa-ppl](mesures/gqa-ppl-2026-11-01.txt)). One prediction missed and is reported beside the decision, not inside it:
a few isolated decode steps move one logit by up to 13.13, where [0.04, 0.5] was predicted. Nothing explains it yet,
and the comparison that would read it, the published path against itself across its two paths, has not been run.

**Two traps.** The first grouped prefill failed on the card and not on the Mac, for $0.09: the first append to a `Cat`
cache returns the value projection as a transposed view, which candle's CUDA matmul refuses and its CPU gemm accepts
(`rows_contiguous`, commit `2f8bdd3`). A green test on a Mac proves nothing about a layout on a card. The second is
older and cost nothing this time: any new `KvStore` variant breaks the exhaustive matches of `fusedrun`'s A2 modes, and
the CUDA half of `llvq-llm` has no compiler on this machine, so the image build is where it shows, 17 minutes later.

**How it is served.** The operator chose a field of the served config over an engine default, on 2026-11-02. An
optional `attention` field names the form; absent means `expanded`, so the six configs that predate it keep their
logits and their digests, and `qwen3-4b-tetra-e4.json` still reads `098fe49c`, the digest the jobs check. A seventh
file, `configs/qwen3-4b-tetra-e4-grouped.json`, is the same object with `"attention": "grouped"`. The store is read
from the form and never written beside it, so no config can name a pair no measurement ran. `shared` is refused by
name. `grouped` is refused by name on Metal, where it has never been measured. `Served::apply` is the one place the
pair reaches a model, and the three served doors call it; `bin/gsm8k` applied `LLVQ_KV_PREALLOC` to its served arm
until then and now refuses that variable beside `LLVQ_CONFIG`.

**The door, measured.** A fifth job, $0.54 against $0.87 announced, ran the served door under each config and then
the context sweep beside each. The grouped config reads 133.3 tok/s against 121.5 after a 5-token prompt, and by the
sweep 134.5, 113.8, 98.0 and 77.5 against 122.5, 82.7, 61.4 and 33.2 at 5, 2,048, 4,096 and 8,192 (*measured*,
[served-grouped](mesures/served-grouped-2026-11-02.txt)). The two 256-token texts are the same bytes, so the grouped
config inherits the reference line `8784ffc4` and gate 4 does not change. The expanded arm reproduced the sweep of
2026-10-30 to every printed digit, in another job and another image, which is what says the served door and the bench
door measure one object. All fourteen predictions landed. The prefill is untouched, 35.02 s to the first token at
8,192 against 34.84, and it is now the largest single item of a long-context answer.

**The 4B now serves the grouped configuration**, decided the same day, and the publication perimeter was split to
allow it. Paper 2 cites three Hub repositories by URL and three file digests, and nothing else: those three cards and
the safetensors one are frozen, and a new repository,
[Qwen3-4B-LLVQ-Tetra-served](https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-Tetra-served), carries the same bytes
under a card that is rewritten whenever a measurement replaces a figure. It went up the same day, commit `159afed4`,
seven files: the two configurations, the sealed file, and the `LICENSE`, `.gitattributes` and `SHA256SUMS` of the
cited repository so the structure matches. The Hub reports the file's digest as `886391a8`, the paper's, and the four
frozen repositories were read back at their prior commits afterwards. Only 267 MB crossed the wire: the Hub already
stored the object.

The switch took on a debt, written into ETAT §2 beside the number: the 4B's 133.3 tok/s is the grouped form, while
its 63.37 MMLU and 82.49 GSM8K are the expanded form's, and no task score has ever run under the grouped one. Its
whole licence is the paired perplexity inside ±0.12 %. Settling it costs about $2 for GSM8K through the kernel and
about $20 for an MMLU census.

## 2026-11-04: the 32B shared-memory wall, lifted in code

`docs/format-noyau.md` §8 said two ways out of the `rot_apply` wall were named and neither designed: an f16 staging of
the rotation, or a split into two kernels. Neither was taken. The rotation now stages in device memory where the
card refuses its shared staging (`rot_apply_dev`), and the int4 matvec stages its activation as f16 bits there
(`tv_q4_hs`). Both are byte-identical to the kernels they replace on the host at every width up to 31,744, and both
are selected only where those kernels are refused, so no served object moves. The f16 staging of the rotation was
dropped because it rounds every stage. No card has run either kernel yet; the identity job is drafted, not stamped.

2026-11-04, later the same day: lot 1 ran both kernels on the L40S. Each returned the host's bytes at every width
checked, 25,600 included, and `rot_apply_dev` took 45.43 µs at 25,600 (*measured*,
[lot1](mesures/lot1-4b-2026-11-04.txt), prereg `proofs/preregistration-bigmodel-2-2026-11-04.md`). The 32B is still
never encoded, and nothing has served it end to end.

## 2026-11-04. The grouped attention at 8B and 14B, and its GSM8K at three sizes

The grouped form serves all three sizes since 2026-11-04, and its GSM8K matches the expanded form's at each.

**GSM8K at 4B.** The grouped config scores 81.96 % (1,081 of 1,319) against 82.49 % for the expanded one, -0.53 pp
[-1.07 ; +0.00], McNemar p = 0.092, 10 discordant problems against it and 3 for it (*measured*,
[gsm8k-grouped](mesures/gsm8k-grouped-2026-11-04.txt)). Rule G1 holds, so the GSM8K half of the 2026-11-02 debt is
settled. The interval touches zero and leans against the grouped form.

**8B and 14B.** The grouped form keeps paired WikiText-2 perplexity inside ±0.12 % on both sealed files. By decode it
reads -0.0048 % [-0.0117 ; +0.0021] at 8B and +0.0023 % [-0.0062 ; +0.0109] at 14B; by prefill it is bit for bit
(*measured*, [gqa-8b](mesures/gqa-8b-2026-11-04.txt), [gqa-14b](mesures/gqa-14b-2026-11-04.txt)). At 8,192 it decodes
×2.156 and ×2.140 faster. A light MMLU of 285 questions picks the same answer on every question under both configs:
65.55 % at 8B, 74.13 % at 14B (*measured*, [served-grouped-8b](mesures/served-grouped-8b-2026-11-04.txt),
[served-grouped-14b](mesures/served-grouped-14b-2026-11-04.txt)). The operator granted the T8 at both sizes the same
day. The served doors read 105.8 and 62.5 tok/s against 98.3 and 58.6 for the expanded configs, with the same
256-token texts. They replace the 95.0 and 57.2 tok/s of 2026-09-25, measured before the trims.

**GSM8K at 8B and 14B**, run in lot 1 that evening: 88.55 % against 88.63 %, -0.08 pp [-0.61 ; +0.46], and 91.96 %
against 92.04 %, -0.08 pp [-0.22 ; +0.07], both p = 1.000 (*measured*,
[gsm8k-grouped-8b](mesures/gsm8k-grouped-8b-2026-11-04.txt),
[gsm8k-grouped-14b](mesures/gsm8k-grouped-14b-2026-11-04.txt)).

## 2026-11-04 and 05. A split-KV attention kernel, served from position 1,280

On the morning of 2026-11-05 the served 4B took its own decode-attention kernel from position 1,280, with the grouped
arithmetic below. 4E replaced that config the same afternoon.

**First readings.** The kernel returns the CPU replay's bits on every call checked. It reads the cache at η 0.406 of
864 GB/s at 8,192, where FlashInfer's decode kernel reads 0.810 on the same card (*measured*,
[attn-bench](mesures/attn-bench-2026-11-04.txt), [fi-decode-anchor](mesures/fi-decode-anchor-2026-11-04.txt)). In
the served 4B it gave 95.8 tok/s against 78.0 at 8,192 and ×0.93 after a 5-token prompt. A revised fold raised η to
0.545 and the 8,192 reading to 104.2 against 77.7, and left the short loss at ×0.9408 (*measured*,
[lot1](mesures/lot1-4b-2026-11-04.txt)). Its T8 control did not run: its condition at 5 tokens failed.

**The threshold.** Rows below position 1,280 now take the grouped arithmetic, bit for bit. The kernel config reads
134.7 against 135.0 tok/s after 5 tokens with the same tokens, and 104.0 against 77.5 at 8,192, ×1.3419 (*measured*,
[attn-t8-r2](mesures/attn-t8-r2-2026-11-04.txt)). Paired perplexity on the positions the kernel takes reads -0.0013 %
[-0.0074 ; +0.0047] at 4,096 and +0.0001 % [-0.0076 ; +0.0077] at 8,192. The operator granted that T8 on 2026-11-05,
and `configs/qwen3-4b-tetra-e4-kernel-from1280.json` became the 4B's served config that morning. Its door read 133.7
tok/s (*measured*, [served-4e](mesures/served-4e-2026-11-05.txt)).

**Below the threshold.** With every decode row on the kernel, paired perplexity reads +0.0318 % [-0.0151 ; +0.0786] on
32 windows of 1,280 and +0.0018 % [-0.0053 ; +0.0088] on 12 windows of 4,096 (*measured*,
[attn-t8-below-4b](mesures/attn-t8-below-4b-2026-11-05.txt)). The point at 1,280 missed its prediction. The operator
extended the T8 below 1,280 at 4B only, the licence 4E needs for a short prompt. No task score exists past 1,280,
because GSM8K never reaches that position.

**Beside it.** Lot 1's token-boundary gestures read ×1.0120 [1.0082-1.0128] together with the same tokens, and none
is served under its prereg's rules (lot1). `tv_tetra48_seg` returned the separate launches' bits on 72 of 72 groups
on the card ([seg-tetra](mesures/seg-tetra-2026-11-04.txt)). One more row on the batched path costs 3.93 ms at position
0 ([tk-sweep](mesures/tk-sweep-4b-2026-11-04.txt)).

## 2026-11-04 and 05. The persistent kernel 4E serves the 4B at 178.0 tok/s

The 4B decodes one token a cooperative launch since 2026-11-05. Its door reads 178.0 tok/s, against 133.3 for the
grouped config of 2026-11-02 (*measured*, [served-v4](mesures/served-v4-2026-11-05.txt),
[served-grouped](mesures/served-grouped-2026-11-02.txt)). The operator's target was 300.

**Route P.** One launch over the 4B's 252 matrices is bit-exact and takes 2.394 ms, against 3.009 for the 144 fused
launches, streaming 410 GB/s (*measured*, [pers-l1](mesures/pers-l1-2026-11-04.txt)). Rule H projected 222 to 258
tok/s for 4E from that rate (*estimated*). The rate came from a bench at 32 warps a SM, and 4E runs 16
([revue-debit](archive/revue-debit-2026-11-05.md) §6).

**First 4E readings.** The first was void: a timing round decoded different tokens
([pers-4e](mesures/pers-4e-2026-11-05.txt)). A stale argmax key was the cause (`3e186a0`). The retry decoded the same
tokens at ×0.993 of the eager path ([pers-4e-r2](mesures/pers-4e-r2-2026-11-05.txt)). Phase stamps then showed 4E's
own attention phase taking 2,355 µs of a short token and 72 % of a token at 8,192
([pers-4e-r3](mesures/pers-4e-r3-2026-11-05.txt)).

**v2 and v3.** v2 put attention on the split-KV kernel's mapping and prefetched in the row loop. It reads 178.7
against 132.1 tok/s at 5 tokens, ×1.353, and 131.8 against 103.5 at 8,192, ×1.273, same tokens (*measured*,
[pers-4e-v2](mesures/pers-4e-v2-2026-11-05.txt)). v3 read 179.2 and 135.4
([pers-4e-v3](mesures/pers-4e-v3-2026-11-05.txt)). Alone, v2's row loop streams 379 GB/s; inside 4E the same rows move
about 260 (*measured* and *computed*, [pers-rowloop](mesures/pers-rowloop-2026-11-05.txt)). The served door under the
v3 config read 169.0 tok/s against 133.7 for the served kernel config, with the same tokens at every length and 40 of
50 GSM8K pilot problems right under both (*measured*, [served-4e](mesures/served-4e-2026-11-05.txt)).

**The shaker, and v4 served.** One job timed v3, v4 and v5 with inner stamps. v4 over v3 reads ×1.060 [1.059-1.062]
at 5 tokens, 189.8 tok/s in the measurement form. v5 adds ×1.001 and stays a measurement, refused by name in a config
(*measured*, [shaker-4e](mesures/shaker-4e-2026-11-05.txt)). The operator served v4 on 2026-11-05 through
`configs/qwen3-4b-tetra-e4-4e-v4.json`. Its door reads 178.0 [178.0-178.3] tok/s in 1.38 GB. Its sweep reads 180.8,
169.5, 162.1 and 139.5 at 5, 2,048, 4,096 and 8,192, with the first token at 35.04 s at 8,192 (*measured*,
[served-v4](mesures/served-v4-2026-11-05.txt)). Its text equals v3's everywhere, and every prediction landed. No full
task census has run under 4E, and the 8B and 14B have never run it.

**v6.** Folding ln1, ln2 and the MLP down input into one round trip a thread gives 212.6 tok/s against 190.2 for v4,
×1.118 [1.117-1.118], and 152.0 against 138.9 at 8,192, same tokens (*measured*,
[pers-4e-v6](mesures/pers-4e-v6-2026-11-05.txt)). The down input fell from 796.0 to 254.6 µs a token and missed its
200 µs gate, so by its prereg v6 is not proposed.

The card work of 2026-11-04 and 05 took 25 jobs and $21.27. The registry reads $273.30 over 262 priced rows of 267
(*computed*, `docs/data/jobs.csv`).

## 2026-11-05 and 06. The Metal night: the sealed 4B fused on the Mac, then MLX

The sealed 4B decodes fused on the M3 Max at 52.07 tok/s with its one retained row-loop port, against 7.81 for the
dense reconstruction, with the same 256 tokens (*measured*, M2 of [metal-nuit](mesures/metal-nuit-2026-11-06.txt)).
That ratio is raw: the dense arm copies its f16 head table every token, and Metal has no same-head arm. Everything ran
locally at $0, and nothing is served on Metal.

**The int4 wall.** The fused Metal path was refused at load, because 12 int4 `down_proj` records exceed d_in 8,192.
They now route to a tiled int4 kernel at 2,048 columns a tile, and the served Metal kernels keep their bytes. All 27
mutants of the new kernels and host code were killed before the stamp.

**Attention forms.** On Metal the grouped form returned the expanded form's logits bit for bit on 49,140 decode steps
over 12 windows of 4,096 (*measured*, M5). It decodes ×1.750 faster at 2,048 and ×2.380 at 4,096 (M4). The Metal port
of the split-KV kernel reads ×0.864 after 5 tokens and a wash at 2,048 and 4,096, with paired perplexity +0.0051 % in
its T8 (M6). The Metal door still refuses both forms. Two NOT SERVED configs wait for the operator.

**Prefill.** Two prefill kernels cut the first token of a 4,096-token prompt from 59.73 s to 37.24 s, then from 34.56 s
to 31.53 s, with the same tokens (*measured*, M9 and M10). The two steps chain to about ×1.74 (*computed*). A decode
token is GPU-bound: the host takes 5.43 ms of 15.18 (M12).

**MLX.** mlx-lm's Qwen3-4B 4-bit decodes 123.27 tok/s against our 65.49 after a 5-token prompt (M14). The operator
chose MLX as the Mac's engine on 2026-11-06. A prototype runs our Tetra kernel inside mlx-lm's Qwen3. It decodes 93.92
tok/s after 5 tokens and 70.26 at 4,096, and reads a 4,096-token prompt in 4.11 s at a 3.20 GB peak (*measured*, M16).
mlx-lm reads 129.45, 98.38 and 3.96 s at 3.67 GB, at about 4.5 b/param against our 2.7. The MLX path is not bit-exact
with the Rust engine. On the 5-token house prompt its 256 tokens differ from the Rust engine's in a degenerate
repetition (M15). No perplexity has been read through it.

## 2026-11-07. 4E v6 written as a config for the 4B

Until 2026-11-07 the living state read 4E v6 as a measurement against its missed gate. Its lever left the MLP down
input at 254.6 µs against a prereg gate of 200, so by that prereg's rule v6 was not proposed for serving
([pers-4e-v6](mesures/pers-4e-v6-2026-11-05.txt)). On 2026-11-07 the operator gave the go to implement v6 on the 4B
regardless. `configs/qwen3-4b-tetra-e4-4e-v6.json` is the v4 config with `"rows": "v6"`, and `served.rs` accepts it.
v4 stays served until the door of `ops/jobs/served-v6.sh` reads.

The door read the same day for $0.41 (job 6aeeea70, 825 s of card). It gives 197.2 tok/s against 178.0 for v4, and
150.9 against 139.5 at 8,192, with v4's tokens at the door and every length and v4's 50 pilot completions
([served-v6](mesures/served-v6-2026-11-07.txt)). Every prediction held. Serving it is the operator's call.

## 2026-11-07. plan-300's step 1: no row lever

`docs/plan-300.md` was written on 2026-11-07 with 300 tok/s at the door reachable at the bit on its high bound, 306 to
322 tok/s, if a row lever existed. Step 1 ran the same day on the operator's go (job 6aef03fa, $0.25). No arm on the
Tetra row loop reached the ×1.15 bar; the best, a 128-bit tile layout, read ×1.002 over v2. By the prereg's rule B,
300 at the bit is out on every bound the plan anchors. The int4 tile layout read ×1.338 over 4E's int4 row and is a
lever; no counter beat the grid barrier by 0.3 µs ([step1-micro](mesures/step1-micro-2026-11-07.txt)). 4E v7 (K2) and
v8 (K3a) are coded and bit-exact on the host, not run.

## 2026-11-07. plan-300's step 2: v8 runs, v7 does not load

Step 2 ran the same day on the operator's go (job 6aef3f69, $0.22). v8, v7's staged constants plus a three-stage
cp.async ring in attention, kept the eager tokens and read ×1.076 over v6 at 5 prompt tokens and ×1.025 at 8,192,
227.9 against 211.8 tok/s in the measurement form. v7 alone did not install at 512 threads under the card's ptxas.
K2's gate, read on v8, held: the down input fell from 255 to 150 µs. K3a's gate failed: attention's slowest CTA over
mean reads 4.40, and the job spread is what remains. v8 stays a measurement
([step2-4e](mesures/step2-4e-2026-11-07.txt)).

## 2026-11-07. plan-300's step 3: the int4 tiles hold in situ, the route stops short of 300 at the bit

Step 3 ran the same day on the operator's go (job 6aef51d3, $0.28). v9, v8 with the int4 rows on step 1's tiled
words, kept the eager tokens and read ×1.035 over v8 and ×1.118 over v6 at 5 prompt tokens, 238.6 tok/s in the
measurement form, about 220 at the door (estimated). v6 under v8's shared memory read ×1.000: v8's slower rows on its
stamps were its stamp symbol's spill. The leads left at the bit give about 235 to 258 tok/s at the door (estimated);
300 needs speculation or fewer bytes ([step3-4e](mesures/step3-4e-2026-11-07.txt)).

## 2026-11-07. 4E v9 with one copy of the int4 bytes, the 8B and the 14B named out

The operator asked for v9 on main at 4B, 8B and 14B, with one copy of the int4 bytes. Step 3's measurement form had
held them twice: the served words for the eager path, a tiled copy for 4E, 0.385 GB more on the 4B. Under a config
whose rows are `v9` the loader now tiles the words on the host and uploads them once. The eager path reads them
through `tv_q4_ht` / `tv_q4_hht`, which keep every product and its order; only the addresses move. On the host they
return the served readers' bytes at every int4 width of the three sealed files, and 9 mutants of the tiled reader die.
Each load checks the tiled reader on the card against the served one on a file-order copy, then frees the copy. The
file and its b/param do not move. `configs/qwen3-4b-tetra-e4-4e-v9.json` is written; its door job,
`ops/jobs/served-v9.sh`, is prepared and not run.

4E did not hold on the other two sealed files at first. The 8B holds `o_proj` in Tetra and an untied head; the 14B fits
v9's shared memory (96,064 B of 101,376) and its int4 down rows, but its head is untied too. The operator asked the
same evening for all three sizes. 4E gained the head's own pointers and a Tetra o phase, the MLP's Tetra down at the
context's width; on the host the token loop with both equals the served chain for v4, v6, v8 and v9. The 8B and 14B
v9 configs and their doors, which also read the kernel form's perplexity band against the grouped form, are ready. While wiring this, a test of the v6 branch named `llvq_cuda` without the macOS-or-cuda gate
of pjmalandrino/llvq#12; it would have broken the Linux CI at the merge, and it carries the gate now.
`ops/check-linux-cuda.sh` type-checks the cuda half of `llvq-llm` on the Mac, which no tool did before.

## 2026-11-07. v9 served at 4B, 8B and 14B

The three doors ran the same evening on one image, for $2.65 (jobs 6aef7864, 6aef7ae2, 6aef7ae4). Every gate held:
4E returned the eager path's tokens at every length at each size, the 4B kept the v6 config's text and held the int4
words once (1.980 GB under v9 and under v6), and the kernel form from position 0 stayed inside the perplexity band
against the grouped form at 8B and 14B. At the door v9 reads 221.5, 142.5 and 86.3 tok/s against 197.4 for v6 and
105.9 and 62.8 for the grouped configs, in the same jobs ([served-v9](mesures/served-v9-2026-11-07.txt),
[-8b](mesures/served-v9-8b-2026-11-07.txt), [-14b](mesures/served-v9-14b-2026-11-07.txt)). On the operator's message
the v9 configs became the served ones at all three sizes. Before, the 4B served v4 at 178.0 tok/s and the 8B and 14B
their grouped configs at 105.8 and 62.5.

## 2026-11-07. The MLX door, on the operator's ruling of a tolerance on Metal

The operator ruled open decision 1 of the Metal plan on 2026-11-07: « Sur métal, on a validé mlx donc implem le
aussi ». Until then the plan recorded no ruling on tolerance against bit for bit on Metal, and the M15 prereg had
written "served under the M6 tolerance" as if it were ruled.

The prototype became a door. `ops/mlx_tetra/serve.py` reads `configs/mlx/qwen3-4b-tetra-e4-mlx.json` and refuses by
name every value it does not carry, the Rust keys, any other object, another mlx or mlx-lm, and any `LLVQ_*`
variable. `check.py` gained a reference outside MLX, the `llvq-tetra` decoder that matches Rust bit for bit, and on
the mini fixture a whole forward pass against `transformers`. Before, it compared the MLX kernels only with each
other, so a defect in the shared shaders could pass it. A first mutation run killed 22 of 23 mutants (*measured*). The survivor
gave k the gains of q in the stacked dispatch, invisible on a fixture whose records share their centroids; the check
now scales k's gains apart, and the second run killed 23 of 23 (*measured*, [mlx-door](mesures/mlx-door-2026-11-07.txt)).

The T8 reading ran the same morning on the operator's go, 16.5 minutes, $0. The pooled perplexity read +0.0078 % by
decode and +0.0133 % by prefill against the Rust Metal path, but windows 6 and 7 fell outside +-0.12 % by both paths,
at +0.27 % and -0.13 % (*measured*, [mlx-t8](mesures/mlx-t8-2026-11-07.txt)). By its prereg the door is not served. The same two
windows move by about 0.2 % between two Rust engines, so the gate itself went back to the operator.

The operator chose to keep MLX under a gate on the paired interval, stamped before a reading of windows 13 to 24,
which the first reading had not seen. Against the Rust dense reconstruction in f16 on Metal, the door read
-0.0103 % [-0.0847 ; +0.0642] by decode and -0.0099 % [-0.0842 ; +0.0644] by prefill, inside +-0.12 %
(*measured*, [mlx-pooled](mesures/mlx-pooled-2026-11-07.txt)). The MLX door has served the 4B on the Mac since that reading.

Both readings ran `configs/mlx/qwen3-4b-tetra-e4-mlx.json` at sha256 `d7bd5b5c…`, whose `note`, printed at every
launch, named the per-window gate that failed. A review before the merge rewrote the note alone to name the paired
gate that passed: sha256 `8542b7f9…`, every key the door reads unchanged. The same review moved every argument check
of `serve.py` before the model loads, including the reference file of a reading, and added four refusals (26 of 26
mutants killed, *measured*,
[mlx-door](mesures/mlx-door-2026-11-07.txt)).

## 2026-11-06 and 07. The `llvq` runtime, and a faster CI

`llvq-runtime` came to main (pjmalandrino/llvq#10). The `llvq` binary pulls the three sealed files against pinned
digests and serves them with Ollama's and OpenAI's routes, an OpenAPI document and a chat page. Since
pjmalandrino/llvq#15 it loads the v9 configs on CUDA, greedy only; on Metal it runs the expanded config. CI runs four
jobs in parallel in about 8 minutes, tests under a `ci` profile without LTO (pjmalandrino/llvq#12, #13).

## 2026-11-07. The documentation made light

The four HTML courses (`cours-comprendre-llvq`, `cours-tetra`, `cours-layouts-runtime`, `architecture-c4`) were
replaced by one Markdown document, `docs/how-it-works.md`. Removed with them: the two 27B plans of 2026-09-28, never
adopted; `plan-debit-noyau.md`, whose open leads `plan-300.md` and `ROADMAP.md` now carry; `modele-erreur.md`
(2026-09-15), `inference-cost-reduction-2026.md` (July 2026) and `runbook-rownorms-2026-09-22.md`. `ETAT.md`,
`ROADMAP.md`, `METHODE.md`, `STYLE.md`, `README.md` and `CLAUDE.md` were rewritten shorter on the state of the day.
`AGENTS.md`, a stale copy of `CLAUDE.md`, now points to it.
On the operator's go, 31 entries of `docs/archive/` that nothing outside the archive cites were removed: hand-overs,
drafts, posts and plans of July to November, 33 files and 6,765 lines.

## 2026-11-07. The runtime audit: eight bugs fixed, the text work 350 times cheaper at 8k

The audit of `llvq-runtime` fixed eight bugs, each with a test that failed first, and killed 12 of 12 mutants
([runtime-audit-2026-11](runtime-audit-2026-11.md)). A busy model stopped or deleted stayed in memory out of the cap,
and a second copy could load beside it. A request abandoned in the queue cut the live conversation's cache. Two pulls
of one file could corrupt it, and `llvq pull --from` hard-linked a source that a re-seal could change after its check.
The engine decoded the whole answer at every token, 679 µs a token at 7,936 tokens; the incremental decode takes 1.9 µs
for the same text (*measured*, [runtime-overhead](mesures/runtime-overhead-2026-11-07.txt)).

## 2026-11-07. A CPU backend for the runtime, in place of the dense reconstruction

Asked the same day whether a laptop without a dedicated GPU runs the models. Until then such a machine took
`sealed::load` on the CPU, which rebuilds every projection as f16. Measured for the first time on the 4B: 0.71 tok/s
in a 10.6 GB peak, candle's f16 matvec on one core (*measured*, [cpu-tetra](mesures/cpu-tetra-2026-11-07.txt)).

`llvq-llm` gained `tetra_cpu.rs` and `fused_cpu.rs`. The Tetra word is decoded through three lookups in a 128 KiB
table built from `llvq_search::tetra`'s public objects, not from the GPU kernels, and the activation is put in trio
order once a call. NEON on aarch64, AVX2 + FMA on x86_64, scalar otherwise, rows spread over a rayon pool of
`LLVQ_THREADS`. The int4 records and the int4 embedding are read the same way. `served::load_resolved` sends
`Device::Cpu` there in every build, and the runtime's `Backend::Dense` became `Backend::Cpu`, with a `cpu` config
per entry of `library.json`. Its tests hold the matvec within 2e-6 of `decode_matrix`'s product; 10 of 10 mutants
killed.

At f16 the first reading moved one 4,096-token window by -0.47 % of perplexity against the dense f16, past every
gap between two Rust engines. The same window at f32 read -0.0017 % against the dense f32: the matvec was right, and
candle's CPU f16 arithmetic around it was not. The CPU path runs at f32, which also decodes faster there: 8.6 tok/s at
12 threads and 5.1 at 4, in 1.86 GB, under the bench lock the Mac's sessions share, and -0.0021 % [-0.0022 ; -0.0019] of the dense f32 over 12 windows
(*measured*). Nothing ran on x86, and Windows was never built.

## 2026-11-07. A batching branch: the batched decode of the 4B and its tensor-core kernel

The operator opened a batching axis on 2026-11-07. AWQ in vLLM was measured first at B = 1 to 32 on the L40S: 203.0
to 5,128.7 tok/s aggregate, 541.9 to 427.9 GB/s (*measured*, [batch-awq](mesures/batch-awq-2026-11-07.txt), $0.08).

The branch added a batched step (B sequences, one cache slot each, one row each a step), a batched attention that is
the served kernel's body with one sequence per grid z, and `tv_batch_mma.cu`: Tetra blocks and int4 nibbles decoded
once into shared memory as exact integers, multiplied on the tensor cores, the rotated activation in two f16 halves.
Host probes held the kernel to a double reference (16 of 16 mutants killed) and the attention to the one-row bits (5
of 5). One job read both arms ($0.48): the existing multi-row path plateaus at 231.9 tok/s, the tensor-core kernel
reaches 1,256.5 at B = 32, both with the batch-1 tokens on every row (*measured*,
[batch-tetra](mesures/batch-tetra-2026-11-07.txt)). The gate passed. The kernel's step costs 15.6 ms at B = 1 against
AWQ's 4.9 in vLLM, which leaves Tetra below AWQ in aggregate: 1,256.5 tok/s against 5,128.7 at B = 32, 54.2 GB/s
against 427.9 (*measured*, same journal).

## 2026-11-08. Batching for the L40S: a staged GEMM, B to 128, continuous batching in the runtime

The operator made batching the project's priority on 2026-11-08 ("régler le batch pour défoncer la L40S"). The first
kernel read each weight row's block from global memory one lane a row, with no prefetch. `tv_batch_gemm.cu` stages
the words of a K chunk in shared memory two chunks ahead, folds every scale into f16 weights, runs one MMA pass, takes
up to 128 rows, and launches q, k, v and gate, up as one launch each (17 of 17 mutants killed on the host). The
attention gained a slot-indexed entry, and `llvq serve` a pool of slots under `LLVQ_BATCH`: a request joins a free
slot and leaves without stopping the others.

Two jobs ($1.48 of a $5 envelope) read the gate and the bench at three sizes. The GEMM is x1.67 to x1.81 the first
kernel, keeps every door row equal to batch 1 and perplexity within +-0.0053 %, and scores 40/50 on the GSM8K pilot.
It stays about two times below AWQ in aggregate at every size and B: 3,996.7 against 10,146.2 tok/s on the 4B at
B = 128 (*measured*, [batch-l40s](mesures/batch-l40s-2026-11-08.txt)). The 4B step costs 9.2 ms at B = 1, about 5 ms
of it eager launches, and grows 0.2 ms a row against AWQ's 0.06.

Three more jobs the same day ($1.99; branch $3.47 of $5). The activation in f16 and 128-row tiles gained 12 to 41 % at
B = 128 ([batch-l40s-v2](mesures/batch-l40s-v2-2026-11-08.txt)). The operator then asked for a lone request on 4E and
for two CTAs a SM. The GEMM learned to read 4E v9's tiled int4 words, so the served v9 configs carry the batch: alone,
a request decodes at 224.4 / 142.4 / 88.1 tok/s on the 4B / 8B / 14B, against 106 / 74 / 44 on the GEMM
([batch-l40s-v3](mesures/batch-l40s-v3-2026-11-08.txt)). Two CTAs a SM lost 5 to 17 % and the tiled order about 1 %,
read apart in one job ([batch-l40s-v4](mesures/batch-l40s-v4-2026-11-08.txt)); the occupancy lead is closed.

## 2026-11-07. Speculative decoding with the 0.6B: the bits hold, the speed does not

- Operator's go: test the speculative decode with the 0.6B. Qwen3-0.6B sealed in Tetra on the Mac: 0.252 GB,
  3.099 b/param, perplexity 40.30 against 18.93 for its f16 checkpoint ([draft-0p6b](mesures/draft-0p6b-2026-11-07.txt)).
  `seal` no longer carries a head the model ties to its embedding.
- `pers_4ev.cu`, 4E's verify pass: up to 4 rows of the decode in one launch, each row the served chain's bits on the
  host (11 mutants killed). `speculative.rs`: greedy acceptance, both caches rewound. `bin/specrun`, the gate.
- On the L40S ([spec-gate](mesures/spec-gate-2026-11-07.txt), $2.37 with a first run void at the drafter's load):
  every speculative arm made the greedy tokens at the three sizes, four prompts, k = 1 to 3, and on the 50 GSM8K
  pilot problems. Decode at the best k, x0.595 (4B), x0.629 (8B), x0.625 (14B) of greedy; +0.24 to +0.27 GB on the
  card. A 2-row verify pass costs 1.95 to 2.25 greedy tokens (it spills, runs the prologues and attention per row),
  the drafter 2.5 ms a launch. Not proposed for serving.

## 2026-11-08. The llama.cpp duel: IQ2_S at our bits, three sizes, one L40S

- Prereg stamped before the first job; four deviations, each written before the run it touched: our side on the
  batching Space (D1), run 1 stopped at its first bench by a VRAM line in the JSON (D2), a variable that never reached
  the retry (D3), and a `llama-server` slot starved at its default batch, reproduced on the Mac (D4).
- llama.cpp's nearest rung to our bits is IQ2_S at all three sizes. Against it Tetra wins perplexity and MMLU at
  every size (+9.08, +4.13, +2.89 points) and GSM8K at 4B and 8B; the 14B GSM8K is not separated. IQ2_S decodes 33.5
  to 63.9 tok/s faster after 5 tokens, and wins peak memory everywhere but the 4B after 5 tokens. unsloth's
  UD-IQ2_XXS loops under greedy decoding at 4B (9.17 GSM8K). $17.55
  ([duel-iq2](mesures/duel-iq2-2026-11-08.txt)).

## 2026-11-08. The prefill's card memory at 8,192 tokens: three score tensors, cut to one bit for bit

- The duel against llama.cpp IQ2_S measured 11.48 / 12.66 / 17.09 GB of card memory after an 8,192-token prompt at
  4B / 8B / 14B, against 3.52 / 4.71 / 6.90. The excess was the grouped prefill keeping three f16 score tensors alive
  (6.44 GB at 4B, *computed*), and the stream-ordered pool keeping the first chunk's under the second (*estimated*).
  That model lands on the three measures ([pic-vram-8k](mesures/pic-vram-8k-2026-11-08.txt)).
- The served path now writes the masked softmax back in place, builds the causal mask in f16, syncs the stream
  before each prefill chunk and after each cache growth, and frees the attention's locals before the MLP. The same
  logits bit for bit on the host and on Metal at the real 4B shape, 56.65 to 14.79 GB there (*measured*). On the
  L40S ($2.04): 5.95 / 7.39 / 10.54 GB, inside the computed 5.65 / 7.09 / 10.08 +-0.5, and the logits of the in-place
  softmax equal the whole tensor's (*measured*).
- `"prefill_attention": "kernel"`, NOT SERVED, sends the prompt rows to the decode-attention kernel: no score tensor.
  On the card 4.77 / 7.15 / 9.30 GB against 3.67 / 5.20 / 7.81 computed, 1.14 to 1.21 times slower to the first
  token, its logits the same in chunks of 512 and of 4,096. Not proposed; the excess is not traced.

## 2026-11-08. What the format's maths can buy in throughput

A study at $0 ([`recherche-debit-maths.md`](recherche-debit-maths.md)). The eager Tetra loop's time is linear in its
SASS count a block on the four F1 floor arms, within 2.5 % (*computed*, [sass-tetra-int8](mesures/sass-tetra-int8-2026-11-08.txt)).
Rows at DRAM speed would cap the doors at 278, 196 and 121 tok/s (*estimated*). The int8 activation with `dp4a` takes
the block from 237 to 152 instructions (*computed*), projected 248, 168 and 103 tok/s on the arithmetic (*estimated*),
not at the bit. Two exact levers give 234, 151 and 90 at most. Nothing ran on a card.

## 2026-11-08. Second pass on the throughput study

The first pass gave in-situ Tetra rows the eager loop's instruction share, 64.3 %, which the stream floor forbids:
bounded by the bytes at 671 to 864 GB/s the share is 36 to 51 % at 4B ([sass-tetra-lean](mesures/sass-tetra-lean-2026-11-08.txt)).
The int8 lever drops from 248 / 168 / 103 to 235 / 154 / 94 tok/s at the central figure, and the decode-free ceilings
from 278 / 196 / 121 to 260 / 174 / 105 (*estimated*). New: the word relabeled at load with the Golay patterns by
linearity (237 to 207 instructions, exact), and an int16 dot through `dp2a` (145, rounding 2.0e-5). The best lever,
both with lane rotation, projects 239 / 157 / 95. Turyn, hexacode, Walsh-Hadamard, table-of-dots and sublattice
decodes are closed by computation.

## 2026-11-08. Breaking the throughput ceiling: the budget of a token

Section 11 of [`recherche-debit-maths.md`](recherche-debit-maths.md). At batch 1 the hard limit is the weight bytes,
628 / 359 / 188 tok/s at 864 GB/s; the rest of the 4B's 4.51 ms door is implementation. The single-token floor of
this engine is about 370 / 222 / 124 tok/s (*estimated*). Past it, only multi-token verification: with B1's integer
dot shared across rows and a one-layer drafter, 454 / 288 / 171 at α = 0.7 (*estimated*), gated on a 2-row verify
under 1.15 greedy tokens (spec-gate measured 1.95). The sealed 4B's Tetra words carry 47.4 bits of entropy of 48
(*computed*, `bin/tetraentropy`): lossless coding above Tetra is closed, and so is self-speculation from the lattice.

## 2026-11-08. The integer-dot verify rows on the card: between the gates

[spec-micro](mesures/spec-micro-2026-11-08.txt), $0.48 over two runs, the first nul on a wiring panic. An extra verify
row with B1's integer dot costs 0.187 of 4E's Tetra row, against 0.381 for a second f32 row (*measured*): BETWEEN the
prereg's 0.15 and 0.26, about 0.34 ms in situ at 4B (*estimated*). The integer dot keeps 91.6 % of the rows identical
to the served ones, the rest one binary16 ulp away. Alone, none of B1's levers clears its bar: i16 x1.143 for x1.15,
lean x1.020, rotation x0.856. Projected doors with a one-layer EAGLE-3 head at α 0.6: 319 / 198 / 121 tok/s; with
prompt lookup at α 0.3 to 0.5: 251-300 / 159-187 / 96-113 (*estimated*, `plan-speculation.md`). Three Apache-2.0
EAGLE-3 heads exist for Qwen3 4B, 8B and 14B, +0.24 / 0.22 / 0.19 b/param in int4 (*computed*). The route goes on only
on the operator's call.

## 2026-11-08. Prompt lookup on the card: the copy-heavy prompts only

[spec-lookup](mesures/spec-lookup-2026-11-08.txt), the 4B alone under the operator's $0.75 cap, $0.49. Prompt lookup
keeps the greedy tokens on six prompts and 50 GSM8K problems. Tokens a verify at k = 3: 2.56 on a code edit and 3.04 on
a retrieval answer (α 0.71 and 0.82), 1.18 to 1.51 on chat, code, summary and GSM8K (α 0.15 to 0.35) (*measured*,
α *computed*). On today's verify the decode stays under greedy except on retrieval (x1.012). With the integer-dot
verify the copy-heavy prompts would reach 395 and 469 tok/s at 4B, free text about greedy's door (*estimated*).

The 8B and 14B the same day ($1.76 of the operator's $2.60): identity at both sizes. Tokens a verify at k = 3 on the
code edit 2.54 and 1.98, on the retrieval answer 3.52 and 2.61, 1.09 to 1.48 on free text (*measured*). Projected with
the integer-dot verify: 241 and 334 tok/s at 8B, 115 and 150 at 14B on those two prompts (*estimated*).

## 2026-11-08 to 11-09. The speculative night: closed on today's verify, for $1.04

The operator's go: finish speculation and its tests within $15. One card job and one Mac bench were enough.

- [eagle-alpha](mesures/eagle-alpha-2026-11-08.txt), $0. The Apache-2.0 head `AngelSlim/Qwen3-4B_eagle3`, on the
  sealed 4B through the MLX door, replayed on the greedy text. α 0.43 to 0.61 on free text, 0.67 to 0.81 on GSM8K and
  copies (*measured* / *computed*). On today's verify it projects x0.72 on chat (*computed*). The conditional card job
  J3 was not run.
- [spec-final](mesures/spec-final-2026-11-08.txt), $1.04. Identity at 4B, 8B and 14B on every prompt, phase launch and
  GSM8K completion. An extra verify row costs 0.74 / 0.89 / 0.98 of a 4E token, with 92 to 95 % in qkv, o, gateup and
  down; attention and RoPE take under 0.5 % (*measured*). Prompt lookup with no-draft rounds on 4E and a trigram probe
  policy reads x0.94 / 0.96 / 0.92 / 0.88 of greedy decode on chat, code, summary and GSM8K at 4B, x1.108 on
  retrieval; the house prompt reaches 250.4 tok/s at the door (*measured*). Rule B is not met: lookup closes.
- B1 was cut for the night: the integer dot changes the logits, so it needs a new served object (4E's single-token
  path and the verify, perplexity and MMLU gates at three sizes), weeks of work, not a test.
- The served decode stays greedy 4E v9. The next lever is the K-row loop, on the operator's go.

## 2026-11-09. Draft trees on the Mac: step A of the tree verify holds, $0

[eagle-tree](mesures/eagle-tree-2026-11-09.txt). The same head and greedy text as eagle-alpha, drafting EAGLE-2 trees
instead of a chain: 16 rows emit 2.28 tokens a pass on chat and 3.15 on GSM8K at 3 head steps, against 1.69 and 2.42
for the chain of 3 (*measured*). A tensor-core verify with each prologue run once a row would cost 1.27 [1.03-1.60]
greedy tokens at 16 rows (*estimated*), and the 4B would decode about 352 tok/s on chat [275-435] (*computed*). The
plan is [`plan-verify-tree.md`](plan-verify-tree.md); its first card job, about $0.30, is drafted and not
timestamped. The int4 head's run was interrupted when the Mac froze and was dropped: a quantized head's acceptance
stays unmeasured.

## 2026-11-09. Tree verify S0: the kernel passes its host gates, $0

[tree-verify-s0-host](mesures/tree-verify-s0-host-2026-11-09.txt). `pers_4et.cu` verifies a tree of up to 32 rows in
one launch: projections on the tensor cores with d_in slices folded in order, prologues once a row, the row's keys in
logical order, the accepted rows' K and V moved in the next launch. On the Mac, through `host_grid.h` and an MMA
modelled from the PTX fragments: rows keep their bits at R = 1 to 32 and under slot permutations, a tree row equals
the root it becomes, R = 1 sits within 1.1e-3 to 3.5e-3 relative RMS of 4E's logits with its argmax, and 10 mutants
of 10 die (*measured*). The host fixture's SiLU table became SiLU itself for the closeness gate. A fold-order mutant
first survived: the units were dealt slice after slice, so the last slice always arrived last; they are now dealt
block after block. Nothing has run on a card; S1 still needs its Rust mirror, loader and bench.

## 2026-11-09. Tree verify S1 on the card: the rows are nearly free, the pass is not; the route stops, $0.12

[verify-tree-s1](mesures/verify-tree-s1-2026-11-09.txt). One L40S job (6af20dab, 4 min billed, registry $307.25), on
the operator's go "Oui lance s1", launched from the session "Tetra décodage et vitesse de sortie". At the chat context
an extra row costs 0.0196 of a 4E token, inside its signed interval, but a one-row pass costs 2.80 4E tokens against a
signed 1.09, c_8 2.92 and c_16 3.09 (*measured*). The prereg's rule reads FAIL (c_16 > 1.60 and c_8 > 1.48): the route
stops. Identity holds and the one-row pass makes 4E's token at 64 of 64 positions. The bench stopped at 4E's stamped
token, which `p4e_phase_report` refuses below two tokens, so the rag context and the phase tables are missing
(deviation 1); a second context can only raise the larger value the rule reads. Where the pass's fixed cost goes is
not measured; the tree symbol's 736 B of local memory a thread and its 3 extra grid waits a layer are suspects by
reading. A rerun with two stamped tokens would locate it, and needs the operator's go.

## 2026-11-10 to 11-11. The tree pass repaired, the speculative loop set aside, the chain prefill read in place; nothing served changes, $13.96

On the operator's go of 2026-11-09 and its $25 cap: 27 card jobs, $13.96 of the cap, registry $321.21 (*computed*).
The milestones are in [jalons-debit](mesures/jalons-debit-2026-11-11.txt).

- Tree verify pass, N1 to N11 ([verify-tree-n11](mesures/verify-tree-n11-2026-11-10.txt)). A 16-row pass costs 1.73
  tokens of 4E at the chat context, from 3.07 (*measured*). Identity holds on every arm.
- Speculative loop, SP2 to SP8 ([spec-loop-sp8](mesures/spec-loop-sp8-2026-11-11.txt)). With the DFlash drafter the
  4B decodes 502 tok/s on GSM8K-200 (x2.30), 343 on the house set (x1.56), 242 in chat (x1.09); GSM8K 167 right
  against 166, p = 1 (*measured*). It peaks at 5.41 GB against 2.01 (*measured*,
  [spec-mem-m3](mesures/spec-mem-m3-2026-11-11.txt)), at 4.87 b/param with the f16 drafter (*computed*,
  [spec-memoire](mesures/spec-memoire-2026-11-11.txt)). The operator set it aside as a fast profile; the default stays
  at 2.73 b/param.
- Chain prefill: the prompt as known tokens through the tree pass, reading the served words in place, no copy. First
  token at 1,024 / 4,096 / 8,191 tokens: eager 3.951 / 17.641 / 37.037 s, 16 rows a pass 0.702 / 4.317 / 13.368, 32
  rows 0.449 / 2.720 / 7.679, 32 rows with the full attention tile 0.420 / 2.393 / 6.485 (*measured*,
  [large-prefill](mesures/large-prefill-2026-11-11.txt), [large-prefill-ft](mesures/large-prefill-ft-2026-11-11.txt)).
  The 32-row forms equal the 16-row form bit for bit (argmax, logits, K and V); paired perplexity of the 16-row form
  +0.0007 % against eager. A bench, not served.
- Refuted or failed: E1, 4E's L2 prefetch removed or moved, x1.003 and x0.978 ([large-4e-e1](mesures/large-4e-e1-2026-11-11.txt));
  D1, one K/V read for 4 rows, x1.302 at 8,191 for a x1.5 gate ([large-prefill-d1](mesures/large-prefill-d1-2026-11-11.txt));
  v9ft, the full tile in 4E, x1.004 at 8,192 for x1.05 ([large-attn-ft-4e](mesures/large-attn-ft-4e-2026-11-11.txt)).
- Nothing served changes: no config, no served module, no Hub file. 4E v9 stays the served decode.

## 2026-11-12. The sealed 4B loses its quality in the MLP; CERVEAU and MODES adopted; nothing served changes, $5.26

[ablation-qualite-4b](mesures/ablation-qualite-4b-2026-11-12.txt). One L40S job (6af550a4, 175.4 min) restored
projection groups of the sealed 4B to f16 from the checkpoint, in eight dense arms on the full MMLU split. It cost
$5.26 against $5.79 estimated (*computed*). Spend under the $25 cap of 2026-11-09 is $19.22, registry $326.47. The
rows of `docs/data/jobs.csv` sum to $326.22, $0.25 under the registry; the journal traces the gap to
[large-prefill-ft](mesures/large-prefill-ft-2026-11-11.txt), not chased.

- The MLP carries the gap. Restored to f16, gate/up recovers +3.38 [+2.75, +4.02] MMLU points, down +2.01 [+1.46,
  +2.54], the whole MLP +4.86 [+4.16, +5.57]; attention recovers +0.73 [+0.16, +1.29], qkv +0.55, o -0.13
  (*measured*). All 252 projections recover +6.72 and sit +0.04 [-0.23, +0.31] under FP16. All exceeds the MLP plus
  attention by +1.14 [+0.43, +1.85] (*computed*). Six controls are green.
- Nine of fourteen signed predictions are in. The five misses are all on attention or additivity: qkv, o and attn
  came in under their intervals, A_attn and S above theirs.
- 4E off the bit is a study, not a measurement:
  [large-4e-hors-poids-etude](mesures/large-4e-hors-poids-etude-2026-11-12.txt). Its best composed set takes the chat
  door from 224 to 241-281 tok/s (*estimated*), under x1.3 (291). Nothing was launched.
- The night of 2026-11-11 ran three rounds of ideation, and every idea ended under its bar (+3.4 MMLU, x1.3 of
  decode). The ideas were judged by estimates, not by a measurement or a verifier. The notes are
  `gains-criteres-*`, `recherche-*`, `innovation-*` and `refutation-k1-*` of 2026-11-11 in [`mesures/`](mesures/),
  with [innovation-tetra-v2-forme](mesures/innovation-tetra-v2-forme-2026-11-11.txt). The study notes
  [innovation-ecart-qualite](mesures/innovation-ecart-qualite-2026-11-12.txt) and
  [innovation-cout-encodage](mesures/innovation-cout-encodage-2026-11-12.txt) followed on 2026-11-12.
- [`CERVEAU.md`](CERVEAU.md) now holds what we believe about the subject, 59 claims each with its standing, and
  [`MODES.md`](MODES.md) the two working modes that replace the night's method. Operator's ruling: a per-model tweak
  and a law drawn from a few points are not ideas. Delegation became a standing rule ([`METHODE.md`](METHODE.md) §8).
- An independent review of the same day added the standing `proved` to `CERVEAU.md` and regraded nine claims
  against the definitions: 13 established, 2 proved, 35 supported, 6 conjecture, 3 refuted. The ablation journal now
  states both readings of the prereg's "interacting" row: to the letter it fires and the next step is global.
- Nothing served changes: no config, no served module, no Hub file.

## 2026-11-13. A second research phase opens, on quality; MODES allows two phases on two axes; $0

The operator signed the mandate of a research phase on model quality, O12 of [`CERVEAU.md`](CERVEAU.md). It starts
from the anomaly N1 and runs beside the phase on decode speed (O11). Its budget is five days and $10 of card, under
a cap of its own. Nothing is measured and nothing was launched.

- [`MODES.md`](MODES.md) said until this day: "At most one research phase and one development phase run at a time:
  one card budget, one work machine." The operator replaced it. Two research phases now run together when each
  mandate carries its own card budget, the second loads no model on the work machine, and the two edit no file in
  common but `CERVEAU.md`.
- R0 of the phase is done the same day, by reading only: what the repository knows
  ([recherche-qualite-r0-depot](mesures/recherche-qualite-r0-depot-2026-11-13.txt)) and what is published
  ([recherche-qualite-r0-litterature](mesures/recherche-qualite-r0-litterature-2026-11-13.txt)). N1's Gaussian arm
  is noise on the four option logits of a dump, not on weights, and its replay on the sealed 4B needs dumps the
  repository holds. No published study answers N1 or N2. Eight claims of `CERVEAU.md` gained scope notes.
- R1's first measurement ran the same day under a stamped prereg (sha256 3323f313), on dumps, with no model:
  [recherche-qualite-n1](mesures/recherche-qualite-n1-2026-11-13.txt). N1 holds on the sealed 4B: 63.37 measured
  against 64.65 predicted by Gaussian noise of the same amplitude on the four option logits, −1.27 [−1.89, −0.65]
  (*computed* on *measured* dumps). The amplitude alone predicts 5.49 of the 6.77 points to FP16. AWQ 4B reads −0.88
  on the same split, so the gap is not LLVQ-only. Ten of thirteen signed predictions are in. `CERVEAU.md` Q7 is
  restated: its former sentence, "where it lands matters more than its size", does not hold as shares on this object.
- The operator then moved the phase's starting point from N1 to the amplitude share, the 5.49 points: where an error
  lands against chance cannot be known in advance. N1 is set aside. No budget is added yet.
- On the operator's go under $5, one card job, $0.44 (6af6f9da, prereg sha256 208751f9):
  [layer-vs-output-4b](mesures/layer-vs-output-4b-2026-11-13.txt). Restoring one projection group to FP16 in the
  sealed 4B recovers 36.29 % of the KL to FP16 for gate/up, 23.17 % for down, 8.74 % for qkv and 2.48 % for o
  (*measured*), the order of the MMLU restores. The relative layer-output error reads down > qkv > o > gate/up: it
  rates gate/up lowest. Rule D green, perplexity within 0.033 % of the candle path. Eight of seventeen signed
  predictions are in. Spend of the phase: $0.44 of $10; the rows of `docs/data/jobs.csv` sum to $326.66 (*computed*).
- The operator gave the go for a first re-encode of the 4B with gate/up judged on the final output, in its smallest
  form (gradient-weighted input Hessians, $1.65 to $1.89 *estimated*), and lifted the Mac clause of the mandate for
  its two encodes. Nothing is coded or launched yet.
- Nothing served changes: no config, no served module, no Hub file.
