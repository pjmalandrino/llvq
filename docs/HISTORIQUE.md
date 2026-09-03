# History

The project's chronological thread, one entry per period, from 2026-07-24 to 2026-09-02. The current state is in ETAT.md, the lab rules in METHODE.md, shipped with this file; until they are in `docs/`, [CLAUDE.md](../CLAUDE.md) is authoritative.

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
- The prereg rule has a hole: line 1 requires CI > 1.5 (bound 1.47), lines 2-3 require G4 < 3.0. Operator decision pending ([../proofs/preregistration-m2b-v4bits-2026-09-02-ECARTS.md](../proofs/preregistration-m2b-v4bits-2026-09-02-ECARTS.md)).
- M1, $0, 12 Mac runs, 0.6B/28 blocks, median and range over 3 seeds: ρ = 1 39.6042 / 4.6214; 0.9 27.0812 / 3.1498; 0.7 27.4944 / 0.6847; 0.5 27.9506 / 2.9771 (*measured*, [mesures/m1-hessienne-shrink-2026-09-02.txt](mesures/m1-hessienne-shrink-2026-09-02.txt)). Control 38.4507 replayed.
- By the prereg rule: ρ* = 0.7, M1 green, the signed kill prediction (ρ* = 1) refuted. On n = 3 the range hangs on one seed; the sign and the order of magnitude hold (−12 ppl, seeds 2-3 from 3.47 down to ≤ 0.54). Q1 adopts ρ ∈ [0.5; 0.9], to be re-estimated; n/N 0.023 against 0.074 at 4B (*computed*).
- M1 deviation: the queue moved to nice 10 at the 5th measurement (CPU 1470%, RSS 1.22 GB), ppl bit-exact; rule `LLVQ_THREADS ≈ ncpu−4` and nice from launch. F1 note: projected retention 88.9-89.6% below the 90.3 kill (*estimated*), F1a counts before coding.
- arXiv submission 7927047 rejected: `paper.pdf` uploaded in place of the sources; `\pdfoutput=1` added to `main.tex` for resubmission (`e721bc5`, git). Preregs timestamped: m2-attribution `71712e60`, m1-hessienne-shrink `5a5e1027`, m2b-v4bits `263ec52a`, anchoring pending. Wave 1: $2.46 spent of $5.
