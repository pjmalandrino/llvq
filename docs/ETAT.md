# Project state as of 2026-09-02

## 1. The project

LLVQ quantizes the weights of an LLM to 2 bits on the Leech lattice, in Rust. The goal is to fit larger models on local
hardware. The repository carries the quantizer, the file format and a fused CUDA kernel that decodes and multiplies
without going back through f16. Three Qwen3 models are sealed and served: 4B, 8B, 14B. The manuscript is on Zenodo (DOI
10.5281/zenodo.22133606). The paper was submitted to arXiv as sources on 2026-09-02 (commit `e721bc5`); the operator
reports it published, and the repository does not carry its identifier yet.

## 2. Served configuration v1

The served configuration is `planes14` + `LLVQ_EMBED=q8` + `LLVQ_ROT_SHARE=1` + `LLVQ_FUSE=1`, at all three sizes, on
L40S.

| size | tok/s [range] | GB on card | b/param whole model | wikitext ppl (× f16) | MMLU micro (f16 → LLVQ) |
|---|---|---|---|---|---|
| 4B | 100.6 [99.9–100.7] | 2.57 | 5.162 | 16.94 (×1.3845) | 70.32 → 55.59 |
| 8B | 75.5 [75.5–75.6] | 5.41 | 5.322 | 10.97 (×1.2201) | 76.08 → 65.52 |
| 14B | 46.8 [46.7–46.8] | 9.40 | 5.106 | 9.49 (×1.1894) | 78.97 → 72.12 |

Throughputs and GB: *measured*, medians of 5 rounds, GB in host byte count
([vague2-fusion-8b-14b-2026-08-31.txt](mesures/vague2-fusion-8b-14b-2026-08-31.txt),
[d1-fusion-servie-2026-08-24.txt](mesures/d1-fusion-servie-2026-08-24.txt)). b/param: *computed* on measured bytes, q8
embedding at 8.5 b/param ([rtbits-14b-2026-08-17.txt](mesures/rtbits-14b-2026-08-17.txt)). Quality: *measured* on the
sealed file, fingerprints `3f1baca9033bf251` and `65dcd53655e8bfa5`
([a4-campagne-2026-08-06.txt](mesures/a4-campagne-2026-08-06.txt),
[campagne-8b-qualite-2026-08-08.txt](mesures/campagne-8b-qualite-2026-08-08.txt),
[campagne-14b-qualite-2026-08-10.txt](mesures/campagne-14b-qualite-2026-08-10.txt)).

The dense f16 path yields 43.5 / 26.4 / 17.0 tok/s in 8.04 / 16.38 / 29.54 GB (*measured*,
[b2-fusedrun-plages-2026-08-18.txt](mesures/b2-fusedrun-plages-2026-08-18.txt)).

## 3. Competitors at 4B

| arm | disk GB | b/param whole model | MMLU micro | ppl (× f16 of its own stack) |
|---|---|---|---|---|
| f16 | 8.04 | 16.0 | 70.32 | ×1 |
| AWQ w4 g128, official Qwen | 2.67 | 5.302 | 70.04 | ×1.105 |
| LLVQ 2-bit, `Planes14` + q8 | 1.77 (1.41 in int8) | 5.162 | 55.59 | ×1.3845 |
| IQ2_XXS, llama.cpp Metal | 1.25 (*measured*, 1,246,620,832 B) | 2.479 | 39.39 | ×2.6287 |

Disk: *measured*, ×4.54 over f16 ([fiche-4b.md](fiche-4b.md)). At 4B LLVQ wins disk and memory, loses quality. Neither
the throughput nor the memory of AWQ can be read in our harness: it is dequantized to f16 there. Its b/param holds in
its own engine. AWQ and f16: *measured*, same harness, same fingerprint (a4-campagne). IQ2_XXS: *measured* in its own
stack ([m3-iq2-metal-2026-08-30.txt](mesures/m3-iq2-metal-2026-08-30.txt)); its MMLU crosses engines to within 0.52 pp
(*measured*, m3-iq2-metal), its perplexity does not. Paired gaps: LLVQ loses 14.45 pp [11.60, 17.27] to AWQ (*computed*,
[mmlupair-4b-8b-2026-08-13.txt](mesures/mmlupair-4b-8b-2026-08-13.txt)) and gains 16.20 pp [12.64, 19.72] over IQ2_XXS
(*computed*, m3-iq2-metal). The gap to AWQ is 7.49 pp at 8B and 6.09 pp [3.62, 8.52] at 14B
([mmlupair-14b-2026-08-17.txt](mesures/mmlupair-14b-2026-08-17.txt)). In memory we are below the official AWQ at all
three sizes: −2.6%, −10.6%, −5.5% (*computed*, [rtbits-14b-2026-08-17.txt](mesures/rtbits-14b-2026-08-17.txt)).
Paper reference, Table 6, 4B without fine-tuning: LLVQ shape-gain with 0 gain bits 17.05 ppl and 60.7% MMLU, QTIP
17.04 and 57.4 (*measured* by the paper, [llvq-paper-notes.md](llvq-paper-notes.md)). In excess log-likelihood we are
2.6% worse than QTIP (0.3254 against 0.3171 nats, f16 on the sealed file, *computed*,
[fiche-4b.md](fiche-4b.md) §3.1); the shortfall of 5.1 pp against the paper's 60.7 is unexplained. Against us:
131k calibration tokens versus 6,100 sequences, and the input rotation alone.

## 4. Structural facts

The served recipe is Algorithm 1 (shape-gain, gain reset) plus an incoherence rotation on the input. The retraction of
Eq. 17 is a no-op under a coded gain and Algorithm 3 (`group_scales`) is disabled; "Spherical GPTQ" names the
`llvq-quant` crate, not the recipe ([fiche-4b.md](fiche-4b.md) §2.3).

The format unfolds 4.804 b/weight in VRAM (*measured* on the bench,
[e2-golay70-bench-2026-08-07.txt](mesures/e2-golay70-bench-2026-08-07.txt)) for 2.1595 b/weight written in the sealed
file, tail included, over 3,633,315,840 projection weights (*computed*, `bin/seal`).

The `nullk` floor belongs to our launch geometry, not to the card. It is 2.306 ms for 252 projections without reading a
weight, 4.77× f16 (*measured*, [f2-p3-qtip-banc-2026-08-21.txt](mesures/f2-p3-qtip-banc-2026-08-21.txt)).
QTIP finishes the same projections in 2.246 ms, at 4.89× [4.89–4.90], reading 0.91 GB. `Planes14` takes 5.103 ms for
2.18 GB; the ratio is 2.27× [2.27–2.28], close to the traffic ratio of 2.40×. The comparable quantity is GB/s
(405 against 428). These × are L40S: on A100 none of our arms beats f16 (*measured*,
[f4-a100-2026-08-18.txt](mesures/f4-a100-2026-08-18.txt)).

The calibration-window draw carries σ = 5.2% in perplexity over three full 4B runs: 16.7425 / 15.8836 /
15.1027 (*measured*, [f5-graines-4b-2026-08-19.txt](mesures/f5-graines-4b-2026-08-19.txt)), range 10.3% (*computed*).
In MMLU it carries 2.92 pp (range 5.83 pp; *measured*,
[bruit-mmlu-graines-4b-2026-08-25.txt](mesures/bruit-mmlu-graines-4b-2026-08-25.txt)). The scaling curve
compares objects calibrated identically: the 4B, 8B and 14B artifacts all ran without a seed, on the same contiguous
prefix of 131,072 tokens of C4 shard 00000 (*measured*, fiche-4b). Each absolute level is that of a single
draw. A second seed at 8B and at 14B is missing. The published file (another shard) is not a fourth draw. Any
effect that recalibrates is read against this σ. For an A/B at constant file the bar is the paired interval, ±0.12% in
ppl and 0.43 pp in MMLU (*measured*, [kvq8-4b-2026-08-15.txt](mesures/kvq8-4b-2026-08-15.txt)).

Same-head, the kernel gain grows with size: ×1.11, ×1.29, ×1.41 from 4B to 14B (*measured*,
b2-fusedrun-plages). The raw series (×2.00, ×2.57, ×2.55) has no order; it dates from ROT_SHARE=0/FUSE=0, never
replayed under v1.

## 5. Results of 2026-09-02

| batch | cost | what is measured | result |
|---|---|---|---|
| M1 | $0, 12 Mac runs, 0.6B 28 blocks | shrink `H ← ρH + (1−ρ)diag H`, 3 seeds | ρ=1 median 39.6042, range 4.6214; ρ=0.9 27.0812 / 3.1498; ρ=0.7 27.4944 / 0.6847; ρ=0.5 27.9506 / 2.9771 |
| M2 | ~$2.17, 72.3 min | MMLU 4B, each projection type restored to f16, constant file, 11 arms | gate +5.18, up +4.94, v +4.48, down +2.96, o +2.35, k +2.09, q +1.85 pp; attention +6.90, MLP +10.78, all +14.73 |
| M2b | ~$0.29, ~10 min | `v_proj` in int4 g128 dequantized | MMLU 59.19, +3.60 pp [1.47, 5.79], McNemar 2.0e-4; 5.149 b/param |

Results *measured*: [m1-hessienne-shrink-2026-09-02.txt](mesures/m1-hessienne-shrink-2026-09-02.txt),
[m2-attribution-4b-2026-09-02.txt](mesures/m2-attribution-4b-2026-09-02.txt),
[m2b-v4bits-2026-09-02.txt](mesures/m2b-v4bits-2026-09-02.txt). Durations and costs *computed* on the timestamps of the
bucket ($1.80/h). Preregs timestamped before each job, Bitcoin anchoring pending. Wave 1: $2.46 spent out of 5.

M2 points at `v_proj` (2.6% of the weights, *computed*, m2-attribution); the "k_proj and attention" prior is refuted.
Serving `v_proj` in f16 would cost +0.263 b/param (5.425, above AWQ). In int4 g128 it gives back −0.013 (5.149)
(*computed*, m2b-v4bits). The unfolded Leech weighs 4.804 b/weight where int4 g128 weighs 4.250 (*computed*,
m2b-v4bits). M2b keeps 80.4% of the f16 gain and brings the gap to AWQ from 14.45 down to 10.85 pp (*computed* on M2
and M2b). M1 is green: the predicted kill (ρ* = 1) is refuted; at n = 3 the reliable part is the sign and the order of
magnitude (median −12 ppl). Q1 adopts a ρ in [0.5, 0.9], to be re-estimated at 4B (n/N 0.074 against 0.023 at the
0.6B, *computed*, m1-hessienne-shrink). Knobs shipped: `LLVQ_RESTORE_F16`, `LLVQ_RESTORE_Q4`, `LLVQ_H_SHRINK`.

## 6. Open decisions

- Reading rule for M2b, operator. None of the three lines of the timestamped rule applies: it requires a CI >
  1.5 and the lower bound is 1.47
  ([preregistration-m2b-v4bits-2026-09-02-ECARTS.md](../proofs/preregistration-m2b-v4bits-2026-09-02-ECARTS.md)).
- Push `recherche/m1-m2-vague1` (15 unpushed commits; `main` = `origin/main`), operator.
- Next workstream, operator: mixed precision on `v_proj` (F3 under F1, one codebook per matrix) or F1. F1a must
  count the exact retention before any code: the projection yields 88.9 to 89.6%, below the F1b kill at 90.3%
  (*estimated* on the research branch, [projection-gains-2026-09-01.md](archive/projection-gains-2026-09-01.md)).
- Product triplet in force (operator, 2026-08-16): 8k context, 5 GB margin, 32 GiB unit, offload as reference
  only. It leaves 27.93 GB to the weights, so b_max = 3.00 kernel b/weight; `Planes14` exceeds it by 60%. The largest
  admissible class is 43.3 billion parameters at 5.162 b/param (upper bound, embedding 9.7%) and 45.8 billion at
  4.878 (embedding ~2%). The 32B is the served object, the 70B does not fit (*computed*,
  [note-produit-2026-08-13.md](archive/note-produit-2026-08-13.md) §B bis).
- Replication of M2 on seed 3 of F5, Q1 prereg at 4B, `ots upgrade` of the day's three timestamps after anchoring:
  operator.

## 7. Closed absent a new idea

- E1v on the served path: 0.25× f16 (*measured*, [e1v-cuda-2026-08-16.txt](mesures/e1v-cuda-2026-08-16.txt)).
- `Golay70`: v2 at 1.77× [1.76–1.78], below the timestamped threshold of 2.0× (*measured*,
  [golay70-v2-sept-bras-2026-08-11.txt](mesures/golay70-v2-sept-bras-2026-08-11.txt)).
- E3: 3.0444 kernel b/weight against a criterion of 2.60 (*computed*,
  [radixstudy-x4-2026-08-12.txt](mesures/radixstudy-x4-2026-08-12.txt)).
- Calibration volume: the oracle gives −1.6% of ppl, ×13 the tokens −1.2% (*measured*,
  [verdicts-lot-b-2026-08-06.md](archive/verdicts-lot-b-2026-08-06.md)); the scale-up at 4B never started, MMLU σ
  2.92 pp > 2.0.
- int4 g64 embedding (`q4b-e4.llvq`, 1.211 GB, 2.4093 b/weight): +1.52% of ppl (*measured*,
  [verdicts-lot-b-2026-08-06.md](archive/verdicts-lot-b-2026-08-06.md) §B4); only int8 (−0.02%) is served.
- Design C: ×1.99 of ppl at 28 blocks (*measured*, [verdicts-nuit-2026-08-07.md](archive/verdicts-nuit-2026-08-07.md)).
- `group_scales`: 44.66 → 53.60 of ppl at 28 blocks of the 0.6B (*measured* at the smoke of 2026-07-28, calibration
  131k tokens, no journal).
- A2 served (CUDA Graphs): +12.6% of throughput against +47% of VRAM at 4B for a KV window of 8k (*computed*, never
  measured,
  [preregistration-a2-a3-geometrie-2026-08-31-ECARTS.md](../proofs/preregistration-a2-a3-geometrie-2026-08-31-ECARTS.md)
  §É7). Reopening if the served context drops to 2k (+12% of memory for +12.6% of throughput). It also holds if the
  KV cache moves to q8 or if the capture accepts a cache that grows. `KvStore::Cat` stays the default.
