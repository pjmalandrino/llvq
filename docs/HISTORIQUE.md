# History

The project's chronological thread, one entry per period, from 2026-07-24 to 2026-09-06. The current state is in ETAT.md, the lab rules in METHODE.md, shipped with this file; until they are in `docs/`, [CLAUDE.md](../CLAUDE.md) is authoritative.

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
- MMLU reported in micro, as the paper does. The prediction "drop of ~−10 pp after correction" is refuted on 08-02: the aggregation was worth 0.93 pp (*computed*, [mmlu-micro-2026-08-02.log](mmlu-micro-2026-08-02.log)).
- G4 benchmark re-anchored on `LeechShapeGain` (gain coded on the norm): shape–gain 0 bit 88.90%, MSE 0.0850 (*measured*, `llvq-bench`).
- `bin/thesis` on Metal: FP16 21.69 ms against LLVQ 10.46 ms, 252 projections, 1,105,920 rows checked against f64 (*measured*, [mesures/thesis-temoin-2026-08-04.txt](mesures/thesis-temoin-2026-08-04.txt)).
- The published 2.07× is the top of a range [2.029; 2.080] over three invocations (*measured*, [mesures/thesis-temoin-2026-08-04.txt](mesures/thesis-temoin-2026-08-04.txt)).

## 2026-08-02 to 08-04. MMLU micro, 8B, 32B, single shell

- MMLU micro 4B on Metal: 70.42 ± 1.28 against 56.09 ± 1.36, −14.33 pp (*measured*, [mmlu-micro-2026-08-02.log](mmlu-micro-2026-08-02.log)); baseline +0.22 pp from the paper. Replaced by the L40S measurement of 08-06.
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
- q8 embedding in production: ppl 16.9358, MMLU 55.70 (*measured*, [mesures/campagne-finale-bras4-2026-08-07.txt](mesures/campagne-finale-bras4-2026-08-07.txt)). The journal gives `fusedrun` at 88.4 tok/s in 2.60 GB as displayed; the campaign summary writes 88.4-88.5 (*measured*, [campagne-finale-2026-08-07.md](campagne-finale-2026-08-07.md)), a single point.
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

## 2026-09-04. M2b arbitrated, F1a counted

- Operator decision: the case left uncovered by the M2b rule is read **cashable**. Both axes move at once, +3.60 pp of paired MMLU for −0.013 b/param, and nothing is paid for the gain. Q5 opens ([../proofs/preregistration-m2b-v4bits-2026-09-02-ECARTS.md](../proofs/preregistration-m2b-v4bits-2026-09-02-ECARTS.md) §É4).
- The decision does not repair the rule: line 1 stays failed on its letter, the CI lower bound is 1.47 and never reaches 1.50 over eight bootstrap seeds (*measured*, [mesures/m2b-v4bits-2026-09-02.txt](mesures/m2b-v4bits-2026-09-02.txt)). Carried forward: no kernel serves `v_proj` in four bits, M2b dequantizes it to f16 before the matvec, so Q5's served quality is unmeasured.
- F1a, first exact count: a shaping region that is the product of three 8-dimensional balls gives **89.10%** of retention at 2.000 b/dim against a kill at 90.3% (*computed*, closed form, `ops/f1a_shaping.py`). Sphere shaping gain 0.7292 dB in dimension 8 against 1.0958 in dimension 24, loss 0.3666 dB, +8.81% of MSE on the served 0.077718. The bracket of 88.9-89.6% (*estimated*, 09-02) is superseded.
- Denominator trap avoided: 47 bits for the lattice-point field would give 90.99% and pass the kill. The word is 48 bits for 24 dimensions on both sides. Same error as 2026-08-04, when 92.24% was divided by a fractional rate no file pays.
- What F1a has left: the three sections are chained by the 8 state bits, so the product is a hypothesis. The region must reach 0.8742 dB of shaping gain to clear the kill, 39.6% of the product-to-ball gap, and 0.9585 dB to be adopted at 91.0%, 62.6% of it (*computed*, same script).
- `recherche/m1-m2-vague1` is contained in `origin/main` (*measured*, git); the push decision of 09-02 is closed. The replicate of M2 on a second seed is still open, ~$2.17 of the $2.54 left in the wave-1 cap.
- Replicate of M2, job `6a9a8cc1e686246ca69a0d2d`, 71 min, $2.14: the eleven arms on the seed-3 artifact of F5 (*measured*, [mesures/m2rep-graine3-4b-2026-09-04.txt](mesures/m2rep-graine3-4b-2026-09-04.txt)). Both controls pass harder than asked: the shipped arm's dump is identical byte for byte to `mmlu-s3.csv` (55.17%), and "all restored" gives 70.32%, M2's value to the hundredth.
- `v_proj` retained by the preregistered clause, CI [+1.11; +4.68] overlapping [+2.39; +6.61]. The clause discriminates nothing: all seven types overlap between draws.
- No difference between draws is resolved (largest `gate`, z = −1.68, *computed*, SEs in quadrature), but the head of the ranking swaps: `gate` 1st → 4th (+5.18 → +2.71), `down` 4th → 1st (+2.96 → +5.55). `q` and `k` lose resolution on seed 3 (p = 0.66 and 0.15). Stable across draws: MLP ≫ attention, `v` third, `q`/`k` last.
- The signed prediction of the M2 prereg §6 ("largest isolated Δ on down_proj") is refuted on the published file and confirmed on seed 3, with an unresolved difference between the two. That is draw dependence, measured.
- Consequence: the f16 ceiling of `v_proj` falls to +2.87, so int4 extrapolates to +2.31 pp (*computed*, M2b not replayed), under Q5's gate of +3.0. The cashable reading of 09-04 holds for the published file; its generalization does not. Wave 1 closes at $4.60 of $5.
- Launch defect logged: a first job left without the `HF_TOKEN` secret, cancelled in SCHEDULING before billing (`6a9a8c90e686246ca69a0d24`, $0.00).
- M2b replayed on seed 3, job `6a9abb71259f8e97255de73a`, 15 min, $0.45: `v_proj` in int4 g128 gives 57.87% against 55.17%, **G4 = +2.71 pp [+0.59; +4.93]**, McNemar 0.0106 (*measured*, [mesures/m2b-graine3-4b-2026-09-04.txt](mesures/m2b-graine3-4b-2026-09-04.txt)). The shipped dump is byte-identical to `mmlu-s3.csv` for the third independent job.
- The CI clears zero, so line 1 of the timestamped rule applies: **Q5 is adopted**, the mixed-precision kernel is built, and the served figure is the range +2.71 to +3.60 pp over two draws, never +3.60 alone.
- The signed prediction (+1.8 to +2.8 pp, CI above zero) is right on the number and on the conclusion — a first in this file.
- The survival rate of the f16 gain into int4 is 94.4% on seed 3 against 80.4% on the published file. It is not a constant of the format, and the +2.31 pp extrapolated from it four hours earlier was an artefact. Two points are not a trend; no cause is claimed.
- Cap overrun declared: $0.29 announced, $0.45 spent, wave 1 closes at $5.05 on a $5.00 cap. Nothing launches before a wave-2 cap.
- Q5 kernel started: `llvq-llm/kernels/tv_q4_h.cu` (one warp per row, int4 g128, f32 accumulation) with `tests/proj_q4.rs` — dequantization bit-exact against `RawTensor::to_f32`, row dot against an f64 reference, and a nibble swap proved lethal. Launch geometry and barriers remain an open claim until a card runs it.
- F1a counted, $0 (*computed*, `llvq-bench --bin f1count`, [mesures/f1a-comptes-2026-09-04.txt](mesures/f1a-comptes-2026-09-04.txt)). The natural coordinate order carries 2^10 = 1024 states at each section cut, so the roadmap's 8-bit state field would not fit. An ordering into three disjoint octads — a trio, found among the 759 — brings each cut to **2^8 = 256**, and the 8-bit field fits exactly. Λ₂₄ adds two bits to the Golay state and no more.
- The middle section has 1,024 edges over 256 states, **4 branches per state**: a (state, branch) → 8 coordinates table weighs **8.0 KiB** against the 16 KiB gate. **F1a is green on its stated criterion.** The ~13-bit label splits into 2 coded bits and ~11 arithmetic ones; only the coded part is tabulated. Closing check 512 × 4 × 2 = 4,096, the code's word count.
- Carried: the trio ordering changes the index map, hence `codebook_fingerprint`; format v2 moves up from F1b to F1a. Not done: the bijection is counted, not proved, and the arithmetic bits assume an E₈ rank decode that is uncosted — E1v died on decode cost, not on bytes.
- Correction the same evening, found by an adversarial review of the F1b spec: the branch count above is wrong by a factor of four. The 1,024 edges are Golay edges over **64** Golay states, not 256 Λ₂₄ states — **16 branches per state**, 4 coded label bits. The 8.0 KiB table figure and the "green with 2× margin" verdict are withdrawn; at eight coordinates per entry the table is 32 KiB against a 16 KiB gate. **F1a's table gate is not established**, and no replacement figure is claimed: what a branch entry holds belongs to the F1b spec.
- The closing check passed on the wrong number because the two errors cancel: 512 × 4 = 128 × 16 = 2,048. A check that cannot tell the right count from the wrong one is not a check — same failure as §É1 of the M2 replicate.
- Settled by the same review: **F1b needs no format change.** It is a benchmark, the trio reordering stays inside the new module, `codebook_fingerprint` does not move, and format v2 returns to F1c where the roadmap had it.
- Truncation-rule study, $0, nine agents, four families each adversarially attacked ([archive/f1-regle-de-troncature-2026-09-04.md](archive/f1-regle-de-troncature-2026-09-04.md)). **The decisive result is a ceiling, not a measurement**: any per-section region is a product of three 8-dimensional regions and the best of those is the ball, so no rule of the form `[state][s1][s2][s3]` exceeds 0.7292 dB. The F1b kill needs 0.8742 dB. **F1b cannot pass its own gate, whatever the truncation rule** (*computed*, confirmed by exact enumeration and by `ops/f1a_shaping.py`). Only joint bounding across sections can, and that is F2.
- A rule meeting all three constraints does exist: lowest-norm ±pair table plus a per-state even-weight sign mask, ~250-270 ALU ops per block against `Planes14`'s ~250, 0.6895 dB. Its cost is L1: 1,112 KiB of tables against the 16 KiB gate, or 80 KiB on the odd Leech coset alone.
- Voronoi codes, predicted here as the answer, are dead on decode cost: ~1,018 ops against a ~250 budget, the E8 nearest point alone costing 217 ops per section against the ~20 predicted. The prediction was right on bijectivity and on the 0.65 dB of the E8 Voronoi cell, wrong by an order of magnitude on the decode.
- Two corrections carried: the bijection is now proved offset-independently by two independent constructions, so the covolume caveat is struck; and the F1d gate is mis-anchored on QTIP's grid — a zero-cost decoder would fail it at ~3.50 ms against a 3.369 ms kill. $0 to fix, before any F1d prereg.
- The 89.10% product figure survived two attacks (finite-N granularity corrections of −0.089 and −0.28 dB); enumeration over the real regions puts the correction at −0.02 to −0.06 dB per section, and the same estimator returns −0.0202 dB on the served side, so the corrections largely cancel.
- Three F1 gates repaired, operator go, $0 ([ROADMAP.md](ROADMAP.md) §2.2 bis). F1d fired on a decoder costing nothing — its thresholds were QTIP's times in QTIP's grid against our own floor of 2.306 ms; restated on our floor plus the arm's own traffic. F1a's 16 KiB came with "after factorizing bases × signs", and the study measured that factorization leaving 67 orbits, not one; restated on the card's measured attributes, 99 KiB opt-in per block minus the 12 KiB activation tile. F1b's "kill < 90.3%" was "no worse than shell 12", which was buried for being *strictly dominated* at the same cost — an argument that does not transfer to a format halving the VRAM, and which the 0.7292 dB ceiling makes unsatisfiable anyway.
- The new F1b gate is a retention loss of 3.5 pp against an in-process ball-12 control, anchored on two measurements rather than judgement: 3.5 pp transposes to −1.16 to −2.31 pp of MMLU by the repo's own first-order rule, and Q5 returns +2.71 to +3.60 pp. ⚠️ All three were rewritten after computing that F1 fails them as written; none uses an F1 result as its anchor, and the operator can override.
- Operator rule, 2026-09-04: **a gate is written on a fundamental criterion, never on a proxy** — the four axes (disk, VRAM, throughput, quality) plus loadable model class, encoding cost and noise floor. Recorded in [METHODE.md](METHODE.md) §1. It came from the day's audit: all three unsound F1 gates were proxies (Gaussian retention, table kibibytes, a competitor's milliseconds in its own grid), while the Q and M axes, already written on perplexity, MMLU and b/param, survived the same audit untouched.
- Applied at once to the F axis. F1a and F1b lose their gates: neither can measure a fundamental criterion, so they carry a feasibility blocker and a measurement. The axis's first gate becomes **F1c** on perplexity; F1d reads throughput and VRAM against `Planes14` in our own process rather than QTIP in its grid; F1e reads all four axes at once. The axis has fewer decision points than this morning, all later and all on measured quantities.
- F1b implementation started, $0, Mac: `llvq-bench/src/f1.rs` — trio permutation, the three-section trellis, and the per-section point sets with their counting DP. Eight tests, zero clippy. The strong ones: walking the trellis rebuilds the permuted Golay code set for set (4,096 words, none extra, none missing), a real path through it lands in Λ₂₄ under `llvq_core::Leech::contains` in the repository's own coordinate order, the DP agrees with brute force histogram bin by histogram bin, and the section covolumes are confirmed by counting points in a ball rather than asserted — 2¹⁶ / 2¹² / 2¹⁶, whose product over 256 states is det(√8·Λ₂₄) = 2³⁶.
- **Mutation-tested rather than trusted**, after two checks passed on wrong values earlier the same day. Five mutants: a non-disjoint trio (killed, 8 tests), a state definition dropping `L_future` (killed, 5), a DP pooled to one pattern (killed, 2), a wrong section-3 parity (killed, 1), and a k-range short by four — which **survived**. No test saw it: the brute-force comparison runs at a small `t_max` and the covolume ratio absorbs a thin tail inside its tolerance. Closed with an always-on guard asserting the range spans every `v ≡ base (mod 4)` with `v² ≤ t_max`.
- The first version of that guard was itself too weak: it probed one step outside a range that moves with the mutation, so it moved with it too. Ranges short by one to three were then confirmed **equivalent mutants** by direct enumeration — they lose no value of `v` — so surviving them is correct and not a second hole.
- F1b truncation implemented: exactly `2^w` lowest-norm points per section, ties on the boundary shell broken lexicographically, with the rank computed **per pattern and summed** — never pooled across the patterns of a coset, which an adversarial review measured inflating one real section coset from 2,401 to 609,553, a factor of 254, silently. Eleven tests; the decisive one is that the counted membership test and brute-force enumeration agree on every point on and below the boundary shell, sharing no code beyond `contains`.
- Mutation-tested again, five mutants on the truncation alone: tie-break `<` to `<=`, the lexicographic comparison off by one, parity dropped from the rank, the tie order reversed, and the boundary shell included whole. **All five killed.** Across the module: ten mutants tried, nine killed, one that survived closed with a new guard, and three confirmed equivalent by enumeration.
- Cross-validation on a number, not on the code: the truncation radii come out at ρ² ∈ 88..96 for the end sections and 72..80 for the middle over all 128 state-parity combinations each — **exactly** what the adversarial review derived independently in Python from the same Golay construction. Two implementations sharing no code agree on every bound (`cargo run --release -p llvq-bench --example f1radii`).
- F1b section encoder built: the constrained nearest point of a truncated coset, by the textbook D₈ decode — round every coordinate of `(target − p·1 − 2c)/4`, and where the k-parity is wrong re-round the single least-decided one — plus single-coordinate fallbacks and radial shrinks, over a precomputed region whose membership test costs one comparison for the interior. A guaranteed lowest-norm member makes the encoder infallible: without it it returned nothing for targets outside the region, and "no answer" is not something a quantizer may do.
- The design's known bias **measured rather than assumed**: against exhaustive search over the whole region, the candidate search is exact on **400 of 400** targets in each of the three sections, at a spread matched to the region radius (`cargo run --release -p llvq-bench --example f1enc`). Where it is not exact it can only return a point farther away, so the bias is one-sided and against F1; here it is zero in the regime the scale sweep puts the target in. Fourteen tests on the module, zero clippy.
- F1b encoder complete, $0: the full three-section search over the 256-state trellis. The bench scores shape-gain, `e² = ‖x‖² − 2·g·t + g²` with `t = ⟨x, y⟩/‖y‖` and `g` fixed by `‖x‖` alone, so the objective is purely angular and does **not** decompose across sections. What decomposes is `‖x − s·y‖²` at a fixed scale, hence a scale sweep: solve each section independently at `s`, join through the trellis, score the winner on the true angular objective. Every candidate is feasible by construction, so a coarse sweep can only understate F1.
- Verified before optimised: every block the encoder emits lands in Λ₂₄, decided by `llvq_core::Leech::contains` in the repository's own coordinate order after undoing the trio permutation — 20 of 20 (`cargo run --release -p llvq-bench --example f1time`).
- Structural fact checked rather than believed, and it is what makes the measurement affordable: the 64 states share only **eight** distinct middle-byte sets, eight states each, sixteen bytes each, every one a coset of a 4-dimensional subspace (`examples/f1mids.rs`). An adversarial review had claimed it; this reproduces it independently.
- Throughput 1.473 s/block, then **0.122 s/block** after hoisting the section-3 search out of the innermost loop — it depends on the path only through `(r_out, s16)`, 128 combinations, not through the 2,048 triples that reach them. The mean `t` is identical to four decimals before and after, so the twelvefold gain changed the clock and nothing else. 20,000 blocks now cost ~41 min on the Mac. Fourteen tests, zero clippy.
- The F1 decoder table computed for the served split, replacing the figure withdrawn that morning (*computed*, `examples/f1table.rs`). Under coordinate sign flips — the only isometry the shipped kernel applies for free — the 256 end-section regions fall into **67 orbits** and the 16 middle ones into **9**, reproducing an adversarial review's count from an independent implementation. Coordinates reach |9|, so six bytes packed per entry: 67 × 2¹² + 9 × 2¹⁵ = **3,336 KiB**, **34× over** the card's 101,376 B opt-in with the 12 KiB activation tile included. The 16 KiB gate it was once measured against was never the right bound.
- The consequence is traffic, not capacity: 3.3 MB fits L2's 48 MB, but a model pass over the 4B's 3,633,315,840 projection weights is 151.4 M blocks and **454 M table lookups**, which at one 32-byte sector each is **14.5 GB of table reads against 0.98 GB of weight reads — 15 to 1**. `Planes14` reads 2.18 GB of DRAM and decodes from a 12 KiB L1-resident table. F1 reads 0.45× the DRAM bytes and asks for a table 275× larger. Whether L2 absorbs that is what F1d measures; nothing here has measured L2 bandwidth.

## 2026-09-06. Tetra: the format is named, written, and writes its first file

- The format of lead F1 is named **Tetra** (operator, 2026-09-06): the word has four fields, `[state][s₁][s₂][s₃]`. The entries above say Trio because that is the name the work was done under. The **trio** keeps its own name where it means the mathematics: three disjoint octads of the Golay code, the standard term, `llvq_search::tetra::TRIO`. The rename moved the fingerprint's domain string and with it `PUBLISHED_TETRA_FINGERPRINT`, from `0xebc7_5263_8c8d_b088` to `0x9c30_6008_6c7a_13e6`; it was done the day the first artifact was written, which is the only day it is free.
- Steps 2 and 4 of the plan ([ROADMAP](ROADMAP.md) §2.2 quater): `TetraShapeGain` implements `BlockQuantizer` with the served recipe unchanged, and `smoke`, `seal`, `ppl`, `mmlu` and `export` read and write the format. One departure from `LeechShapeGain`, deliberate: `reproject` re-encodes instead of negating the point, because **the Tetra map is not centrally symmetric** — negating a coordinate swaps its rank inside its pair, and rank 7 has no negative at all. Measured: the map refuses the negation of 106 of 400 Gaussian blocks, all of even parity (*measured*, `llvq-quant/tests/g5_design_c.rs`).
- A code kind **per matrix** landed before any file was sealed, so Q5 can serve `v_proj` in int4 beside Tetra matrices later: each v5 record carries its own kind, the header keeps a default and a `kinds_present` mask, and every reader consults the record.
- **The first Tetra artifact exists** (*measured*, Qwen3-0.6B, 3 blocks, one seed, same corpus, one variable): witness `leech1c12` ppl 20.7935 against Tetra 21.4947, both at an effective rate of **2.1656 b/weight to the fourth decimal**, both verified bit for bit over 47,185,920 weights, both sealed and reopened at exactly their smoke perplexity. The sealed files differ by 100 bytes: 16 of header plus 21 records times 4 of tag. Tetra encodes at **42 s/block against the witness's 64**. The served 4B still reads 16.9415. Three blocks and one seed is a plumbing test, not a quality reading; that is step 5.
- Two review findings fixed the same day. `verify_artifact` proved the weights and never the label: a mislabelled file decoded against its own tag and passed, and **48.9% of Tetra words decode to a point the ball indexer accepts**, so the writer's refusal was a coin flip per block. It now reads the raw record's kind and holds it to the codebook the run was asked for. And `TetraShapeGain::reconstruct` was covered by nothing — two mutants survived the whole workspace — so it is pinned against an independent formula, which kills both.

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

- **The 8B is encoded and measured** ($0.90, three arms, [journal](mesures/tetra-8b-2026-09-06.txt)). Encoded on the Mac in **4 h 41** (468 s/block, 36 blocks), all 6,945,767,424 weights read back bit for bit, sealed at 4,324,244,913 bytes against the published file's 4,324,243,889 — the two weigh the same because the archive writes 48 bits per block either way. All of the difference is in VRAM, at the unfold.
- The campaign is three arms on one L40S, one harness, the same token fingerprints: f16 8.9899 / `Planes14` 10.9682 / **`Tetra` 11.0478** in perplexity, and 65.52 / **61.61** in MMLU micro. The published arm replays its 10.97 and its 65.52, so the harness is intact. The AWQ arm was dropped to keep the job inside the wave-2 cap.
- **The memory holds: 3.0672 b/param against 5.3220, ÷1.735.** It is a worse ratio than the 4B's ÷1.867 for an entirely accountable reason — `Tetra` does not touch the embedding, which is 9.67% of the 4B's parameters and **15.20% of the 8B's** (untied heads). In b/weight kernel the ratio improves instead, ÷2.235 → ÷2.270.
- **The quality does not: both axes turn against `Tetra` between the two sizes.** Perplexity goes from better by 4.64% to **worse by 0.73%**; excess log-likelihood from 0.854× the served format to **1.036×**; MMLU from −2.10 pp to **−3.91 pp**. And the MMLU gap is resolved this time and was not at the 4B: 3.91 pp is ~3.5 paired SE against ~1.5 for the 4B's 2.10. The exact paired interval needs the per-question dumps, which are in the bucket and cost $0 to pair.
- The perplexity read **during encoding** predicted the card's to the fourth decimal: ×1.2287 against ×1.2201 in-process on Metal in f32, ×1.2289 against ×1.2201 on the card in f16. The wrong sign was on screen five hours before the job was launched, and no one read it.
- Three causes are confounded in that gap and nothing separates them: the format, the encoder drift of 2026-08-26, and the fact that the published 8B was quantized **on a card**, where `calib.rs` accumulates AᵀA in f32 on the accelerator, while `Tetra` was encoded on Metal.
- 🚨 **The bench ran without a prereg**, against hard rule 2, which is written without condition. It cannot be repaired — stamping now would attest to bytes written after the measurement — so its numbers are raw facts that **gate nothing**. Any decision they inform needs a fresh prereg before its own measurement. The 8B is outside the publication perimeter, which limits the damage without excusing it.
- Cost of the day: $1.69 over two benches. Wave 2 stands at **$1.72 of $2.00**, project total $99.28.

## 2026-09-05. The table floor, and a $0.01 lesson

- The CUDA image built in 19 min against the 40-70 estimated, and the 399 lines of `f1floorbench` that macOS cannot type-check compiled on the first try — the hand audit against `nullkbench` had found the only two errors it had.
- The job then died in seconds, **$0.01**, on `no embedded copy of f1floor.cu`. `load_sources_many` reads an `include_str!`'d table and the new unit had no arm in it. Invisible to the type checker, invisible to `cuhcheck` (which parsed the file from disk and never asked whether the binary carried it), and invisible to any Mac because the table was gated on Linux.
- Both halves closed rather than the one that bit: the embedded constants and the lookup are **un-gated**, so any platform can see the table, and `bin/cuhcheck` now asserts that every table-shipped unit has its arm *and* is parse-checked here. The guard was verified by removing the arm again — it fires, exit 1.
- Its reverse check found a pre-existing hole in passing: `preflight.cu` was shipped by the table and had never been syntax-checked. It parses, and it is now in the list.
- Second attempt, and the bench's own guard fired for **$0.00**: the L40S carries **96 MiB of L2**, not the 48 MB the design assumed, so the 1 GiB DRAM calibration point was only 10.7× the cache — below the 20× the prereg demands, where a "miss" is a tenth hit. Moved to 4 GiB, 42.7×, and allocated zeroed on the device rather than uploaded, because 4 GiB of host RAM in a container of unknown size is a worse bet than contents that are never read for meaning.
- Two failures, $0.01 total, both caught by something written to catch them. The measurement has not run yet.
- Third attempt, **$0.01**, 18 s of card: D(8 KiB) 0.344 ms, D(16 KiB) 0.663, a flat plateau of 4.52 ms from 128 KiB to 16 MiB, 10.6 at 64 MiB, 61.4 at 4 GiB; Dsm 3.33 / 5.10 ms; Didx −0.103 ms, negative and unexplained (*measured*, [f1-plancher-table-2026-09-05](mesures/f1-plancher-table-2026-09-05.txt)). Signed prediction right on D(4 MiB) and D(16 KiB), wrong on Dsm(48 KiB) (0.5–2 predicted).
- Reported the same morning as "F1 is dead on decode cost, ~3× the budget, and placing the hot set in shared memory makes it worse". **Withdrawn the same day** after an adversarial audit (six independent readings, three counter-verified, four $0 computations on the Mac, 800 to 27,376 blocks encoded). What fell: the bench runs at six blocks per SM, not eight, so L1 is 28 KB with 12 KiB of tile in it and the 16 KiB point already misses; the shared-memory arms confound occupancy and 3.4–6.8 GB of per-block staging with placement, and Dsm compares 8 warps against 48 — the cross-occupancy reading §6 forbids; "worse" is withdrawn (QTIP does 1.82 G shared lookups per pass in 2.246 ms in the same F2 bench). What held: the plateau is a sector throughput, 3.2 TB/s, not a latency; the real access distribution, computed for the first time, puts 17–21% of lookups in the hottest 16 KiB against the 71–77% the budget needs, so the 67/9 table sits at 2.4–2.5× H — a fact about that table.
- Corrections in passing: 452,044,800 lookups per pass, not 454,164,480 (tails); an entry fits in **4 bytes** for every orbit ((y − p)/2 ∈ [−5, 4]), so the table is 2,224 KiB, not 3,336 — `f1table.rs` counted the sign twice; the L40S's L2 is 96 MiB (the 09-04 entry above says 48 MB).
- **Operator rule, 2026-09-05: a kill is written on a fundamental criterion and by the operator alone.** A floor, a bracket or a projection informs it and never pronounces it. Recorded in [METHODE.md](METHODE.md) §1; deviations of the floor prereg in [ECARTS](../proofs/preregistration-f1-plancher-table-2026-09-04-ECARTS.md).
- Out of the audit, a decoder that fits the floor: a **universal rank table**, one 16 KiB table (2 parity classes × 2,048 rank vectors × 4 B) for all 528 regions, exact for p = 1 and a two-profile compromise for p = 0. Measured on the F1b harness, same blocks, same process: 89.05% against 89.69% for exact F1 (2,000 blocks, −0.64 pp), then 88.88% against 89.48% (4,000 blocks, −0.61 pp), ball-12 control 92.00% both times, paired MSE loss 1.7 ± 0.2%, 0 blocks outside Λ₂₄ (*measured*, `examples/f1rankbench.rs`). The signed prediction before the run was a loss of 2.0–4.5 pp. Its table costs 0.34–0.66 ms on the measured curve; the unknown is now the arithmetic decode, where E1v died.
- Also found by the audit, on fundamental criteria the floor never touched: the bench encoder runs at 240 ms/block/core against F1c's 656 µs/block gate, **366×** — a production encoder precedes any F1c; F1e's "≤ 2.6 b/param" was unreachable at 4B by construction (q8 embedding: 2.76), rewritten on the triplet's b_max; under F1 the 70B fits the triplet's 27.93 GB at 19.5 GB (*computed*) while the `rot_apply` wall still closes the served path past 14B.
- The F1b journal, cited by the floor prereg, did not exist: written on the 05 from the raw output kept in the session transcript ([f1b-retention-2026-09-04](mesures/f1b-retention-2026-09-04.txt)), with its deviations (control §4.5 replaced by `f1enc` + a one-sided argument; two splits on 2,000 blocks; a wrong-grid pilot at 64.65%).
- **The universal-table decoder compiled, verified on the card, and measured** ($0.01, 17 s; prereg stamped before, sha256 `119b02d5…`; [journal](mesures/f1-rang-plancher-2026-09-05.txt), [ECARTS](../proofs/preregistration-f1-rang-plancher-2026-09-05-ECARTS.md)). Built by two agents against one spec and reviewed by three (decoder equivalence: 44 mutants on the header killed by the clang++ harness; compile-ability: the Linux half type-checked on the Mac for the first time with `CUDARC_CUDA_VERSION=12040 --target x86_64-unknown-linux-gnu`, now in CLAUDE.md; protocol and registration). On the card: 64,512 blocks decode to the Rust reference's coordinates; 48 registers, 0 local; `T = 3.346 ms` against `B = 2.797` — 1.20×. The signed prediction was wrong three times (S below: 0.645 for 0.9–1.3, a difference not a bandwidth; Du above: 2.701 for 0.9–1.6; T above: 3.346 for 2.0–3.0) and right on registers. The table is 0.66 of Du; the arithmetic and the dependent small-table chain ~2.0 ms — kernel work, not table work. The prereg's row: F1d is written with this decoder, ~95 tok/s projected at the 4B (*estimated*) for 1.36 GB, and the operator weighs it.
- **The production encoder prototyped** ($0, [journal](mesures/f1-encodeur-prototype-2026-09-05.txt)): 99.7% of the bench's 240 ms were 191,488 membership tests per (block, scale); membership in closed form (cost < C or a lexicographic cut, checked over all 8⁸ rank vectors), the bench's rule kept to the letter (307,200 solves, 0 disagreement), and a lazy trellis encoder: **290–296 µs/block/core** in three runs against the 656 µs gate, the bench's points on 6,000 (block, scale) pairs. Scales measured: one costs 1 pp, two adaptive 0.16 pp, three sit at the gate. Its counter-review: the gate is an encoder-only figure (`encbench` 680–709 today), the pruning rate is unmeasured on real residues (exhaustive: 962 µs), α was fixed on evaluation blocks, and class = [cost ≡ 0 mod 16].
- **The format-v2 map drawn and counter-reviewed** ([ROADMAP](ROADMAP.md) §2.2 ter): three crates and the `llvq-llm` wiring, no shader; found on the way — the disk bit order is MSB-first where the F1 word is little-endian (a transcoder, pinned by test), `read_matrix_raw`'s signature has 20 callers (a per-file code kind in the header instead), and **F1c's gate as written has no power at ρ = 1**: the ±1 cross-seed range is ±11.7% of the median while the expected effect is +0.7 to +1.8 ppl (*estimated* from +8.5% of MSE). A paired Δ per seed at a fixed ρ with a signed prediction is proposed; the gate's form is the operator's.
- **The encoder on real blocks, $0** ([journal](mesures/f1-encodeur-blocs-reels-2026-09-05.txt)): 20,000 compensated, rotated blocks of Qwen3-0.6B captured from the served GPTQ loop by a recording `BlockQuantizer` (`llvq-llm/examples/f1recdump.rs`, deterministic dump, sha256 identical on two captures) — **298 µs/block/core**, ratio 1.00 to Gaussian, base-member rate 55.5% against 55.6%, no drift by matrix or column position; retention gap to ball-12 identical (−3.20 against −3.18 pp). The rotation makes the blocks Gaussian (kurtosis 3.01). The encoder's load-bearing assumption is closed.
- **Three arithmetics for the same decoder**, written independently against one spec, host-verified, one bench ($0 so far): V1 builds the float without the int→float pipe and runs `tv_f1r`'s exact FMA chain; V2 proved the three trellis byte maps **linear over F₂ under the current numbering** (k = 0 on all 4,096 paths, 12 mask columns — `examples/f1linear.rs`, pinned by four tests in `f1::rank`) and computes the patterns with no load, so the format does not move; V3 reads the values as bytes through PRMT. Twenty-eight mutants across the three, 26 killed, 2 equivalent. The review caught a real one before any card time: the CUDA intrinsic `__byte_perm` masks its selector to three bits per nibble, so the `prmt` sign-replication mode V3's byte mask relied on is unreachable — it would have decoded wrong on the card while passing the host shim, which models the instruction. The multiply-mask form is now the default. Control 7 of the bench is the prereg's per-row form; a first draft justified an ∞-norm form with a synthetic model that put 539 rows over the tolerance — replayed with the bench's exact arithmetic, the worst row drifts 3.3e-7, and the draft was wrong.
- **The three arithmetics on the card** ($0.00, 4 s of card after 50 min of queue; prereg stamped, sha256 `a3765c0a…`; [journal](mesures/f1-rang-variantes-2026-09-05.txt), [ECARTS](../proofs/preregistration-f1-rang-variantes-2026-09-05-ECARTS.md)): every variant equal to `tv_f1r` on all 30,720 rows (V1 at Δ = 0 exactly, V2 and V3 at 3.16e-7 — the review's host replay had said 3.3e-7), all three at 40 registers and 0 local. **V3: T = 1.694 ms against B = 2.797 — 0.61×; V1 1.719; V2 3.444 (+0.11 on `tv_f1r`).** The 24 int→float conversions per block were ~1.6 of the ~2.0 ms of arithmetic measured at 14:53; the dependent chain of small-table reads cost nothing. The signed prediction was wrong on all four times, in the direction its own instructive clause had named (`Du_v3 < 1.3`). Reproducibility of the reference arms across the two jobs: 0.4%. The prereg's row: F1d takes v3, v1 equivalent within the ±0.1 ms resolution. Projected, same reserves as the morning's floor: ≈ 113 tok/s at the 4B, +12%, for half the VRAM (*estimated*) — where the naive writing of the same decoder projected −5% five hours earlier.
- **Trio, steps 0, 1 and 3, built and reviewed the same night** ($0; three builders, three adversarial reviewers; commit `cc23f9a`). Step 0: `llvq_search::trio`, 1,122 lines, rebuilt from `llvq_core::Golay` with its own construction and asserting at construction the trio, the 64 states, the 1,024 edges, N0 = 1,240, the closed-form bounds and the 12 F₂ columns; `decode` in natural order, `encode` the inverse refusing every non-codeword; held to the bench's decoder on 10⁵ words, 10⁶ round trips, 16 mutants killed. Step 1: the production encoder, allocation-free and `Sync`, the bench's rule to the letter (6,000 of 6,000 pairs at the bench's point), α = 0.3218 and the pair (α, 1.14 α) fixed on the 4,000 training blocks before the evaluation blocks were read once (88.89% against 92.00); **329 µs/block/core** against `nearest_angular`'s 687 in the same process (*measured*, `trioencbench`, reproduced by the reviewer at 328 and 327), gate 656; the truncated rule stays a knob at 246. Step 3: format v5 (`LVQ5`), the kind in the header, the Trio fingerprint beside the untouched v1 one, encode/decode by kind, refusals in every runtime transcoder, in the fused loader and in every tool that would read a Trio index as a Ball class — seven cuda/metal bins the review found unguarded included; the default writer still writes v4 byte for byte. The review's other finding: the per-file kind cannot carry Q5's `v_proj` in int4 beside Trio matrices; a per-matrix kind goes in before any Trio file is sealed.
- Kept from the audit's ten example files, four: `f1accesscv` (real access, held-out), `f1shrink` (orbit counts under signed permutations 67/9 → 6/3, second moments, six-section trellis), `f1rankbench` and `f1rankenc` (the universal table and its encoder's exactness). Operator go, 2026-09-05: journals, the ALU floor of the universal decoder at ≤ $0.10, and the objective **an F1 that can be tested**.
