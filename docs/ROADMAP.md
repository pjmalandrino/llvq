# Roadmap

What comes next for the project, with its gates and its costs. State as of 2026-09-02. The past is
in [`HISTORIQUE.md`](HISTORIQUE.md), the rules in `METHODE.md`.

## 1. Starting point

The published 4B gives 100.6 tok/s in 2.57 GB in served config v1 (*measured*,
[d1-fusion-servie-2026-08-24](mesures/d1-fusion-servie-2026-08-24.txt), freeze at the three sizes in
[vague2-fusion-8b-14b-2026-08-31](mesures/vague2-fusion-8b-14b-2026-08-31.txt)). It loses 14.73 pp of
MMLU against f16 (*measured*, [a4-campagne-2026-08-06](mesures/a4-campagne-2026-08-06.txt)) and
14.45 pp against 4-bit AWQ (*measured*,
[mmlupair-4b-8b-2026-08-13](mesures/mmlupair-4b-8b-2026-08-13.txt)), for 5.162 b/param against 5.302
(*computed* on measured bytes,
[rtbits-planes-8b-2026-08-09](mesures/rtbits-planes-8b-2026-08-09.txt)). A2 (CUDA Graphs) gives
+13.45% at 4B (*measured*, [a2-verdict-2026-09-01](mesures/a2-verdict-2026-09-01.txt)). It is not
served: its KV window costs +47% of VRAM at 8k, +1.21 GB on 2.57 (*computed*,
[preregistration-a2-a3-geometrie-2026-08-31-ECARTS](../proofs/preregistration-a2-a3-geometrie-2026-08-31-ECARTS.md)
§É7).

The kernel does not reopen the product question. At 100% of its byte bound, `Planes14` tops out at
3.33× FP16, that is 16/4.804 (*computed*,
[plan-cloture-2026-08-27](archive/plan-cloture-2026-08-27.md)), under AWQ at 3.38× (*measured*,
[spec-apres-awq-2026-08-10](archive/spec-apres-awq-2026-08-10.md)) and at 0.68× of QTIP (*computed*).
Quality decides what comes next. A lever that closes 4 to 6 pp of MMLU reopens the product
question; without it, the product strand closes on the current conclusion. The research roadmap is
adopted (D0, commit `1e8583c`) with a $5 cap for wave 1, defined as M2 plus a replicate on a second
seed. M2 cost ~$2.17 (*computed* on the bucket timestamps,
[m2-attribution-4b-2026-09-02](mesures/m2-attribution-4b-2026-09-02.txt)); the replicate has not run.

The paper is deposited on Zenodo (concept DOI 10.5281/zenodo.22133606) after TACO returned it without
review on 2026-08-27. On 2026-09-02, a first arXiv submission (7927047) was refused: the PDF had been
uploaded in place of the sources. The sources were resubmitted the same day (`\pdfoutput=1` added,
commit `e721bc5`). Neither an acceptance nor an arXiv identifier is on record.

## 2. Research roadmap

Three axes. M measures and builds tooling, F looks for a format without unfolding, Q looks to lose
less. Every A/B at 0.6B follows the design C gate: 28 blocks, same seed, then 3 seeds. Any experiment
that recalibrates is read against σ = 5.2% of perplexity (*measured*,
[f5-graines-4b-2026-08-19](mesures/f5-graines-4b-2026-08-19.txt)) and 2.92 pp of MMLU (*measured*,
[bruit-mmlu-graines-4b-2026-08-25](mesures/bruit-mmlu-graines-4b-2026-08-25.txt)).

Four leads stay closed, each measured once. They are calibration volume, a format that leaves the ALU unchanged
(`Golay70`, E1c, E3), the decode race on `tv_planes` and the 32B before D4. The $30 cap is confirmed
wave by wave, never as a cumulative total.

### 2.1 Axis M, measurement

| id | lead | cost (*measured* if done, `jobs.csv`; *estimated* otherwise) | adoption | kill | state |
|---|---|---|---|---|---|
| M1 | off-diagonal shrinkage of H, 0.6B, 28 blocks, 3 seeds | $0 | cross-seed range divided by 2, median held | ρ* = 1 | done, green |
| M2 | MMLU attribution by projection type, constant file, 11 arms | ~$2.17 (*computed*) | measurement, reading criterion | none | done |
| M2b | `v_proj` in int4 g128 dequantized, constant file | ~$0.29 (*computed*, [journal](mesures/m2b-v4bits-2026-09-02.txt)) | G4 ≥ 3.0 and CI > 1.5 | G4 < 1.5 | done, rule undecided |
| M3 | attention entropy per layer; MMLU-STEM column in `mmlupair` | $0 | f16/sealed gap > 3 times the cross-window gap | none | to do |
| M4 | tooling against drift | $0 | none, hygiene | none | to do, section 3 |

M1 is green. The shrinkage `H_ρ = ρ·H + (1−ρ)·diag(H)` gives, at ρ = 0.7, a cross-seed range of
0.6847 ppl against 4.6214 at ρ = 1 (*measured*,
[m1-hessienne-shrink-2026-09-02](mesures/m1-hessienne-shrink-2026-09-02.txt)). The median is 27.4944
against 39.6042. ρ = 0.9 gives 27.0812 / 3.1498 and ρ = 0.5 gives 27.9506 / 2.9771. Caveat: on three
seeds the range hangs on a single seed, a different one for each ρ. Prediction on the record: n/N is
0.023 at 0.6B against 0.074 at 4B (*computed*, same journal), so the effect should be larger at 4B.

M2 is delivered. Gains from restoring one projection type in f16, in pp of paired MMLU (*measured*,
[m2-attribution-4b-2026-09-02](mesures/m2-attribution-4b-2026-09-02.txt)):

| projection | `gate` | `up` | `v` | `down` | `o` | `k` | `q` |
|---|---|---|---|---|---|---|---|
| gain | +5.18 | +4.94 | +4.48 | +2.96 | +2.35 | +2.09 | +1.85 |
| CI95 | [3.04; 7.34] | [2.72; 7.17] | [2.39; 6.61] | [0.71; 5.17] | [0.32; 4.32] | [0.34; 3.79] | [0.22; 3.50] |

Attention as a whole gives +6.90, the MLP +10.78, everything +14.73. The two controls reproduce 2,280
picks out of 2,280. The literature prior on `k_proj` is refuted. The target is `v_proj`: 2.6% of the
weights (*computed*, same journal) for +4.48 pp.

M2b is delivered, without a verdict. `v_proj` in int4 g128 gives 59.19% of MMLU, +3.60 pp
[1.47; 5.79], McNemar 2.0e-4, or 80.4% of the f16 gain (*measured*,
[m2b-v4bits-2026-09-02](mesures/m2b-v4bits-2026-09-02.txt)). Memory goes down: 5.149 b/param
(*computed*, same journal). `Planes14` unfolds to 4.804 b/weight (*measured*,
[c1-planesbench-2026-08-06](mesures/c1-planesbench-2026-08-06.txt)). int4 g128 serves the same
content at 4.250 (*computed*, 4 bits plus an f16 scale and bias per group). Serving `v_proj` in f16
would cost +0.263 b/param, that is 5.425 (*computed*, same journal), above AWQ. Line 1 requires a CI
entirely above 1.5; the lower bound is 1.47, and over eight bootstrap seeds it runs from 1.42 to
1.49, never above 1.50 (*measured*, same journal). Lines 2 and 3 require G4 < 3.0
([preregistration-m2b-v4bits-2026-09-02-ECARTS](../proofs/preregistration-m2b-v4bits-2026-09-02-ECARTS.md)).

New knobs: `LLVQ_RESTORE_F16` and `LLVQ_RESTORE_Q4` (`mmlu`, `ppl`, require `LLVQ_MODEL`),
`LLVQ_H_SHRINK` (`smoke`).

### 2.2 Axis F, format without unfolding

F1 codes Λ₂₄ as a three-section E₈ coset code (Forney 1988, Lepowsky-Meurman 1982). The word is 48
bits: `[state 8][s₁ ~13][s₂ ~13][s₃ ~13][gain 1]`. Decoding costs three lookups and two additions.

| id | lead | cost (*measured* if done, `jobs.csv`; *estimated* otherwise) | adoption | kill | state |
|---|---|---|---|---|---|
| F1a | count states and alphabets for 47 bits, prove the bijection | $0, 1 week | tables ≤ 16 KiB per section | over budget, move to F2 | to do, first |
| F1b | codebook in `llvq-bench`, 20,000 blocks, 48 bits packed | $0, 1 week | retention ≥ 91.0% | < 90.3% | to do |
| F1c | format v2, encoder, 0.6B 28 blocks | $0 | ppl within ±1 range of `leech1c12`; encoder ≤ 656 µs/block | out of band on 3 seeds | to do |
| F1d | `tv_l3e8` arm in `planesbench`, QTIP control in the same process | $1 | t ≤ 1.15·t(QTIP); ≤ 2.20 b/weight kernel | t > 1.5·t(QTIP) | to do |
| F1e | 4B sealed in v2, `fusedrun`, paired MMLU | $8 | ≤ 2.6 b/param; MMLU ≥ 55.59 − 2 SE | MMLU < 53% | to do |
| F2 | sequential trellis + trellis shaping, A3 geometry only | like F1 | fallback if F1a or F1b dies | none | not budgeted |
| F3 | per-row cap, 44 to 50 bits/block, guided by M2 | $7 | +2 pp paired MMLU at constant b/param | < +1 pp | after D3, conditional on F1c |

F1b is under its kill from the estimate alone. The projected shaping loss is +7 to +9% of directional
MSE (*estimated*,
[projection-gains-2026-09-01](archive/projection-gains-2026-09-01.md) §1.4), which is a Gaussian
retention of 88.9 to 89.6% (*estimated*, derived from that same +7-9%) against a kill at 90.3%.
F1a must count the exact retention of the shaping region before any line of code. Otherwise F1 stops
at its gate.

The served path freezes the gain field at 1 bit: 8 assertions, 4 shaders,
`llvq-cuda/src/planes14_host.rs:113` refuses any other value (*measured*, grep). The v2 format of
F1c, the per-row cap of F3 and any Q arm that changes the code reopen the runtime layout on top of
the quantizer.

### 2.3 Axis Q, quality

| id | lead | cost (*measured* if done, `jobs.csv`; *estimated* otherwise) | adoption | kill | state |
|---|---|---|---|---|---|
| Q1 | H shrinkage in production, ρ in [0.5; 0.9] | $0 at 0.6B, $7 at 4B | range ÷ 2 held, median ≤ +range | none | to do, opened by M1 |
| Q2 | asymmetric target and output weighting | $0 then $7 | Δppl ≥ 2 ranges, 3 seeds | < 1 range | to do |
| Q3 | beam GPTQ, K in {2, 4, 8} | $0, 10 h Mac | Δppl ≥ 2 ranges, σ not increased, encoder ≤ K times | < 1 range | to do |
| Q4a | cross-layer equi-norm, VQ version | $0 | Δppl ≥ 2 ranges | < 1 range | to do |
| Q4b | 24×24 maps on the activation side, diagonal first | $0 then $7 | diagonal Δppl ≥ 3%; full +2 pp | none | to do, full after Q6c |
| Q5 | mixed precision on `v_proj` | $7 and a kernel | ≥ +3 pp paired for ≤ +0.10 b/weight | < +1.5 pp | quality measured by M2b, kernel to do |
| Q6a | distillation of the format's free parameters, 0 extra bit | $3 | ≥ +3 pp paired | < +1.5 pp | to do, after M3 |
| Q6b | EoRA / RILQ r ≤ 16 | $3 | ≥ +3 pp within ≤ +0.25 b/param | none | after Q6a |
| Q6c | differentiable relaxation of the Leech search | $0 then $7 | T → 0 bit-exact; Q4b full +2 pp | none | after Q3 |
| Q6d | end-to-end KL distillation, PV-tuning | tens of dollars | ≥ +6 pp paired | none | over the cap, explicit go |
| Q7 | corpus composition, DCLM-edu, 3 seeds | $15 | ≥ +3 pp paired STEM | < +1.5 pp | after M1 and M3 |

Q3 pays for the encoder K times. Two leads have never been tried on it: reusing an octad's partition
for its complement (half the even partitions saved, *computed*) and `pulp` SIMD. Pre-seeding is ruled
out, oracle ceiling 1.37× even and 1.07× odd (*measured*, `bin/encbench`, 2026-07-28): the bound is
too loose. The profiler has never been used.

The Q6 and Q7 gates are read in paired MMLU-STEM: perplexity does not see the collapse of reasoning.
Any gain bought back in bytes fits inside a budget set in advance, in b/param over the whole model.

## 3. Debt and hygiene

- Timestamps waiting to be anchored. On the morning of 09-02, 28 timestamps, 20 anchored, 8 with no
  Bitcoin anchor (*measured*, [ots-etat-2026-09-02](mesures/ots-etat-2026-09-02.txt)): m3-gptq2,
  vague2-gel-geometrie, protocole-piles-isolees-v2, the A2/A3 prereg of 08-31 and the four A2 preregs
  of 09-01. Three more since: m2-attribution (71712e60), m1-hessienne-shrink (5a5e1027),
  m2b-v4bits (263ec52a).
- Two timestamps no longer attest their file, 08-10 and 08-11, rewritten by the anonymization pass
  `01fdbe6`. The attested version is unrecoverable.
- The HF bucket has never been inventoried: 69 files, 46.7 GB as of 08-17 (*measured*, `hf buckets
  ls`). An inventory comes before any re-run quote.
- `[workspace.lints.rust] unsafe_code = "forbid"` and `[lints] workspace = true` on the five core
  crates: `#![forbid]` in `lib.rs` does not cover integration tests.
- Host compilation of the `.cuh` files by `clang++` in CI, on the model of
  `llvq-cuda/tests/host_e1v.cpp`. `ci.yml` does not carry it.
- `ops/status.py` to be written: it generates `docs/ETAT.md` (counters from `mesures/`, `jobs.csv`,
  `otsaudit`, served config) and a CI test fails on a stale counter.
- `docs/exp-piles-isolees-2026-08-30/MACHINES.md:50-52` still gives `ROT_SHARE=0 FUSE=0` as the
  published config; to be aligned on v1.
- No tag points at the deposited commit `e21a8bb`; `v0.0.1` (2026-08-26) points at its direct child
  `16c9c8b` and contains it (*measured* on 09-02, `git tag --contains`). Timestamp owed.
- `docs/hf-model-card.md` carries 5.162 b/param since 08-17; the card online on the Hub has not been
  republished since and diverges. Republishing: operator decision.

## 4. On hold

- MoE. Model settled: Qwen3-30B-A3B. A policy for experts below full rank is missing: 31.4% of
  (layer, expert) cells, one dead expert, measured on gpt-oss-20b, a floor for the 30B-A3B
  (*measured*,
  [moe-routing-gptoss20b-2026-08-12](mesures/moe-routing-gptoss20b-2026-08-12.txt)). P2 is worth
  ~$1.4 and P6 ~$69 (*estimated*).
- q8 KV cache at long context. Quality green at short context, +0.049% of ppl and +0.33 pp of MMLU,
  CI containing zero (*measured*, [kvq8-4b-2026-08-15](mesures/kvq8-4b-2026-08-15.txt)). Long-context
  throughput is not measured: the n_new = 1024 series went over its cap, 661 s against 600
  (*measured*, same journal). Reopening only on a benchmark with a resident model.
- Batch M > 1 and prefill. Batch 1 accepted since 08-18, edge regime and sovereignty. Lazy transcoding
  becomes exact again at M ≥ 8 (*computed*,
  [audit-recherche-2026-09-01](archive/audit-recherche-2026-09-01.md)): the optimal format depends on
  M, to be picked up again if prefill is served.
- The k family. `planes14k`, k in {1, 2, 4, 8}, `TILE_BLOCKS_K = 32`, arms `nullk`, `mvkf16`,
  `cublasf16`: not written. The prereg
  [preregistration-p4-2026-08-14](../proofs/preregistration-p4-2026-08-14.md) is not timestamped;
  its §7bis is still to be filled in (two waivers, and whether the 08-16 `nullk` run was a P4 job).
  K2 reads `T(k=8) ≤ 4.80·T(k=1)`. Shared job $0.8 to $1.0, worst case $2.70 (*estimated*),
  `--timeout 90m`. A k verdict does not carry over to interactive throughput (k = 1); a k benchmark
  that ignores `ROT_SHARE`/`FUSE` measures a path that is no longer served.
- 32B point. ~$62 and 11.4 h on `rtx-pro-6000x2` (*estimated* on 621 s per block, *measured* at the
  de-risking below, no journal; $80 budget with margin). Gate to be formulated on the drop in the
  14B → 32B gap, with its z; official 32B AWQ to be checked. The served path is walled there by
  1,024 bytes of shared memory, `down_proj` rotation (*measured*,
  [rot-partagee-14b-2026-08-17](mesures/rot-partagee-14b-2026-08-17.txt)). De-risking of
  2026-08-03: 4 blocks out of 64, bf16, 59 min, $5.43 (*measured*). `faer` peak 70.6 GB host out of
  512 and 77.4 GB VRAM out of 97 at n = 25,600 (*measured*). `verify_artifact` bit for bit on
  1,950,351,360 weights (*measured*). C3 (bf16 loading) is a prerequisite: 131 GB of f32 do not fit
  in 96 GB, otherwise `h200x2` at ~$180 (*computed*).
  Profile: encoder 71.8%, factorization 16.5%, ~1.9 h of Cholesky in n³ (*measured*). The cost per
  weight rises from 4.77e-5 core-s at 8B to 6.36e-5 at 32B (*measured*). The block predicted at
  ~500 s (*estimated*) cost 621 (*measured*). A ×1.5 encoder brings the run down to ~$40
  (*estimated*) and compounds over every later run.

## 5. Decisions awaited

| decision | deadline | default if silent |
|---|---|---|
| M2b: which §5 line applies to G4 = +3.60 [1.47; 5.79] | before any Q5 kernel | no line, Q5 does not open |
| Q1: prereg with "ρ in [0.5; 0.9] to be re-estimated", size and seeds | before the first Q1 run | Q1 stays at 0.6B, 3 seeds |
| wave 2 cap | before the first paid job | no job on a card |
| `ots upgrade` of the eleven pending timestamps | after anchoring | un-upgraded timestamps in the repository |
| format v2, `codebook_fingerprint` changes | at F1b green | F1 stops at the Gaussian benchmark |
| first Q lead to get the 4B run ($7) | D2, mid-October | best Δppl per range at 0.6B |
| Q6d go, over the cap | D4, December | no |
| F1e passed: second paper or revision | D4, December | second paper |
| next venue for the paper | open | preprint only |
| 32B point: budget go and anchored gate | after D4 | not launched |
| document-extraction domain benchmark ([arXiv:2607.08734](https://arxiv.org/abs/2607.08734)) and CSR, never done; CSR blocked upstream, tasks not transcribed | open | not done |

D1 end of September, D2 mid-October, D3 mid-November, D4 December (*estimated*).
