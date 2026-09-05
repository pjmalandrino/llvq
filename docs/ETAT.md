# Project state as of 2026-09-05

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

On 2026-09-04 the operator reads M2b as cashable: both axes move at once, +3.60 pp for −0.013 b/param, and nothing is
paid for the gain. Q5 opens. Line 1 of the timestamped rule stays failed on its letter, and the rule is not repaired
after the fact
([preregistration-m2b-v4bits-2026-09-02-ECARTS.md](../proofs/preregistration-m2b-v4bits-2026-09-02-ECARTS.md) §É4).
Caveat carried by the decision: no kernel serves `v_proj` in four bits. M2b dequantizes it to f16 before the matvec,
so +3.60 pp is the quality of the format, not of a served path; Q5's work is the kernel, and it reopens the runtime
layout.

## 5 bis. F1a, first exact count of 2026-09-04

A shaping region that is the product of three 8-dimensional balls yields **89.10% of retention** at 2.000 b/dim, below
the F1b kill of 90.3% (*computed*, closed form, `ops/f1a_shaping.py`). The sphere shaping gain is 0.7292 dB in
dimension 8 against 1.0958 dB in dimension 24, a loss of 0.3666 dB, that is +8.81% of MSE on the served 0.077718
([fiche-4b.md](fiche-4b.md)). The bracket of 88.9 to 89.6% (*estimated*) is superseded by this single value. The
definition is checked on the way: the same script gives back 92.14% on the served MSE, the number of fiche-4b.

The rate is 2.000 b/dim on both sides. An F1 word is 48 bits for 24 dimensions, exactly like `leech1c12` plus its gain
bit; dividing by the 47 bits of the lattice-point field alone would give 90.99% and repeat the error of 2026-08-04,
when a 92.24% retention was divided by a fractional rate no file pays ([HISTORIQUE.md](HISTORIQUE.md)).

That figure is now the **ceiling, not a floor**. Any per-section region is a product of three 8-dimensional regions,
the best 8-dimensional region is the ball, so no truncation rule of the form `[state][s1][s2][s3]` can exceed
0.7292 dB. The F1b kill needs 0.8742 dB and its adoption 0.9585 dB — **0.145 dB and 0.229 dB above the ceiling**
(*computed*, `ops/f1a_shaping.py`, and independently by exact enumeration over the actual regions,
[f1-regle-de-troncature-2026-09-04.md](archive/f1-regle-de-troncature-2026-09-04.md)). At 39 bits the integer split
costs a further 0.014 dB, giving 88.98%. **F1b cannot pass its own gate, whatever the truncation rule**; only a rule
that bounds the three sections jointly can, and that is F2. To reach the kill the region must
achieve 0.8742 dB of shaping gain, that is 39.6% of the gap between the product and the 24-dimensional ball; to reach
the 91.0% adoption threshold, 0.9585 dB, that is 62.6% of it (*computed*, same closed form).

The counting half of F1a is also done, and it passes (*computed*, `llvq-bench --bin f1count`,
[f1a-comptes-2026-09-04.txt](mesures/f1a-comptes-2026-09-04.txt)). Under the repository's natural coordinate order the
two section cuts carry 2^10 = 1024 states, so the roadmap's 8-bit state field would not fit and the word would grow
from 48 to 50 bits. Under an ordering that splits the 24 coordinates into three disjoint octads — a trio, found among
the 759 octads — each cut carries **2^8 = 256** states and the 8-bit field fits exactly. Λ₂₄ adds two bits to the
code's state and no more: the shared parity, and the running parity of Σk.

The middle section has 1,024 distinct edges over the **64 Golay** states, that is **16 branches per state** and 4
coded label bits. A first run of the count divided those edges by the 256 Λ₂₄ states instead and published 4 branches,
8.0 KiB and a factor of two of margin; that is withdrawn (correction appended to the journal, found by the adversarial
review of the F1b spec). The closing check passed on the wrong number because the two errors cancel exactly:
512 × 4 = 128 × 16 = 2,048.

**The table is now computed, for the split the measurement uses** (*computed*,
`cargo run --release -p llvq-bench --example f1table`). Under the only isometry the shipped kernel applies for free —
a coordinate sign flip — the 256 end-section regions fall into **67 orbits** and the 16 middle ones into **9**, which
is what an adversarial review reported and what this reproduces from an independent implementation. Coordinates reach
|9|, but every coordinate of a section shares the parity p, so an entry stores (y − p)/2 ∈ [−5, 4] in four bits: **4 bytes**,
and the table is 67 × 2¹² + 9 × 2¹⁵ entries = **2,224 KiB** (the six bytes and 3,336 KiB written on 09-04 counted the sign twice,
[HISTORIQUE](HISTORIQUE.md)). Against the card's 101,376 B opt-in, tile included, that is **22× over**. It cannot be in shared
memory, and the 16 KiB gate it was measured against was never the right bound.

It fits L2 (96 MiB, *measured* at the attribute on 2026-09-05), so the question is traffic, not capacity, and the traffic is
the finding. A model pass over the 4B's 3,633,315,840 projection weights is 150.7 M blocks (tail excluded), hence
**452 M table lookups**; at one 32-byte sector each that is **14.5 GB of table reads against 0.98 GB of weight reads —
fifteen to one**. Whether L2 absorbs that was measured on 2026-09-05 (§5 quinquies). `Planes14`, which ships, reads
2.18 GB of DRAM at the 252-projection bench and decodes from a **12 KiB** constant table that is L1-resident. So F1
reads 0.45× the DRAM bytes and asks for a table 275× larger. Whether L2 absorbs that is exactly what F1d measures,
and nothing in this repository has measured L2 bandwidth.

Untouched by the error, and independently rebuilt in Python from the repository's own Golay construction by two of the
three reviews: the state counts, the trio, the 128 prefixes and 128 suffixes, the 1,024 edges, and the fact that
rebuilding the code from the trellis returns the permuted Golay code set for set.

A truncation rule meeting all three constraints does exist, and its cost is L1 rather than ALU: a lowest-norm
plus-minus pair table with a per-state even-weight sign mask runs at ~250-270 ALU ops per block, parity with
`Planes14`'s ~250, and reaches 0.6895 dB. But the region is per-coset data, and the only isometry the shipped kernel
applies for free — coordinate sign flips — leaves 67 orbits on the end-section cosets and 9 on the middle: **1,112 KiB
of tables against F1a's 16 KiB gate**, or 80 KiB restricted to the odd Leech coset. The Voronoi family, which settles
the bijection group-theoretically and needs no table, costs ~1,018 ALU ops against a ~250 budget — 4.1× over, the E8
nearest point alone being 217 ops per section.

Struck by the same work: the bijection is now **proved**, offset-independently, by two independent constructions, and
no longer rests on the covolume argument. And the F1d gate is mis-anchored — its thresholds are QTIP's times in
QTIP's own grid, while our launch floor alone is 2.306 ms, so a zero-cost decoder at the gate's own 2.20 b/weight
ceiling lands at ~3.50 ms against a kill of 3.369 ms. **F1d as written fires on any decoder.** Re-anchoring it on our
own floor is a $0 edit and must happen before its prereg is timestamped.

Also settled: **F1b needs no format change.** It is a benchmark, the trio reordering lives inside the new module, and
`codebook_fingerprint` does not move. Format v2 returns to F1c, where it was. Still open: the bijection is argued from
covolume (2^16 · 2^12 · 2^16 / 256 = 2^36 = det √8·Λ₂₄) rather than proved, and the arithmetic bits assume an E₈ rank
decode that is not costed — E1v died exactly there, on decode cost and not on bytes.

## 5 ter. Replicate of M2 on seed 3, 2026-09-04

$2.14, 71 min, job `6a9a8cc1e686246ca69a0d2d`. The eleven arms replayed on the seed-3 artifact of F5
(*measured*, [m2rep-graine3-4b-2026-09-04.txt](mesures/m2rep-graine3-4b-2026-09-04.txt)). Both controls pass, and
harder than the prereg asked: the shipped arm gives 55.17% and its dump is **identical byte for byte** to
`docs/data/bruit-mmlu-graines/mmlu-s3.csv`, sixteen days and two unrelated jobs apart; "all restored" gives 70.32%,
M2's value to the hundredth.

| type | seed 3 | published file (M2) | z on the difference |
|---|---|---|---|
| `down` | **+5.55** [3.44; 7.75] | +2.96 [0.71; 5.17] | +1.64 |
| `up` | +4.31 [2.29; 6.39] | +4.94 [2.72; 7.17] | −0.41 |
| `v` | +2.87 [1.11; 4.68] | +4.48 [2.39; 6.61] | −1.14 |
| `gate` | +2.71 [0.80; 4.61] | **+5.18** [3.04; 7.34] | −1.68 |
| `o` | +2.08 [0.32; 3.90] | +2.35 [0.32; 4.32] | −0.20 |
| `k` | +1.11 [−0.61; 2.91] | +2.09 [0.34; 3.79] | −0.78 |
| `q` | +0.49 [−1.22; 2.24] | +1.85 [0.22; 3.50] | −1.12 |

`v_proj` is **retained** by the preregistered clause: its CI overlaps M2's on [2.39; 4.68]
([preregistration-m2-attribution-4b-2026-09-02-ECARTS.md](../proofs/preregistration-m2-attribution-4b-2026-09-02-ECARTS.md)
§É4). The clause discriminates nothing: all seven types overlap between draws.

No difference between draws is resolved, the largest being `gate` at z = −1.68 (*computed*, SEs in quadrature). What
moves without being resolved is the head of the ranking: `gate` from 1st to 4th, `down` from 4th to 1st. What holds:
MLP ≫ attention on both draws (z = +0.44 on the difference), `v` third on both, `q` and `k` last on both — and on
seed 3 those two are no longer resolved against zero (p = 0.66 and 0.15).

**Consequence for Q5, superseded the same day by §5 quater.** The f16 ceiling of `v_proj` falls from +4.48 to +2.87;
at M2b's survival rate of 80.4% int4 would give +2.31 pp (*computed*), below Q5's gate of +3.0. That extrapolation
was measured wrong four hours later.

## 5 quater. M2b on seed 3, 2026-09-04: the gain is confirmed, Q5 is adopted

$0.45, 15 min, job `6a9abb71259f8e97255de73a` (*measured*,
[m2b-graine3-4b-2026-09-04.txt](mesures/m2b-graine3-4b-2026-09-04.txt)). `v_proj` in int4 g128 gives 57.87% against
55.17% shipped: **G4 = +2.71 pp [+0.59; +4.93]**, McNemar p = 0.0106. The shipped arm's dump is byte-identical to
`mmlu-s3.csv` for the third independent job in three weeks.

The CI is entirely above zero, so line 1 of the timestamped rule applies: the gain is confirmed on a second draw and
the mixed-precision kernel is built. **The served figure is a range over two draws, +2.71 to +3.60 pp, never the
+3.60 alone.**

The survival rate of the f16 gain into int4 is 94.4% here against 80.4% on the published file — 14 points apart, so
it is not a constant of the format. The +2.31 pp of §5 ter was an artefact of extrapolating on it, not a property of
four bits. Two points are not a trend and no cause is claimed.

⚠️ Cap overrun, declared: $0.29 announced, $0.45 spent. Wave 1 closes at **$5.05 on a $5.00 cap**, 1% over. Nothing
else launches before a wave-2 cap is set.

## 5 quinquies. The decoder-table floor and the universal table, 2026-09-05

The floor ran on L40S for $0.01 after two failed attempts at $0.01 and $0.00 (*measured*,
[f1-plancher-table-2026-09-05](mesures/f1-plancher-table-2026-09-05.txt); deviations in
[ECARTS](../proofs/preregistration-f1-plancher-table-2026-09-04-ECARTS.md)). Three uniform-random lookups per block, in
`nullk`'s geometry, differenced round by round against a hash-only arm: **D(8 KiB) = 0.344 ms, D(16 KiB) = 0.663 ms,
plateau 4.52 ms from 128 KiB to 16 MiB**, 10.6 ms at 64 MiB, 61.4 ms at 4 GiB. The plateau is a throughput — 3.2 TB/s
of 32-byte sectors, 0.28 lookups per cycle per SM — not a latency (*computed*). The signed prediction was right on
D(4 MiB) (4 to 10) and D(16 KiB) (< 1), wrong on the shared-memory arm (0.5 to 2 predicted, 5.097 measured).

The same-day audit — six independent readings, three counter-verified, four $0 computations on the Mac — corrected the
reading in five places. The bench runs at **six blocks per SM, not eight** (1,536 threads per SM), so the carveout is
100 KB and **L1 is 28 KB**, 12 KiB of it taken by the activation tile: the 16 KiB point already misses 7.6% of its
accesses and is not a pure-hit cost. The shared-memory arms confound occupancy (48 → 16 → 8 warps) and per-block staging
(3.4 to 6.8 GB per pass) with placement, and their difference compares 8 warps against 48 — the cross-occupancy reading
[format-noyau](format-noyau.md) §6 forbids, one level up; "placing the hot set is worse" is withdrawn, and QTIP's 1.82 G
shared-memory lookups per pass in 2.246 ms (F2) stand as the counter-example. The real access distribution, which the
prereg left as a bracket, is now **computed**: the hottest 16 KiB serve 17 to 21% of lookups, 48 KiB 34%, 128 KiB 44 to
48% (*computed*, `llvq-bench/examples/f1accesscv.rs`, 27,376 blocks; three other implementations agree within 4 points).
Holding the F1d budget H = 1.538 ms would need 71 to 77% under 16 KiB. For the 67/9 table as designed that gives
**D ≈ 3.6 to 3.9 ms, 2.4 to 2.5× H** (*estimated*, linear mixing of the measured points). That is a fact about that
table. It is not a kill: the prereg's own §1 says the floor decides nothing, and the operator's rule of 2026-09-05
([METHODE](METHODE.md) §1) is that a kill is written on a fundamental criterion by the operator alone.

What the audit produced instead is a decoder that fits the floor. A **universal rank table** — every section reads one
shared table of 2 parity classes × 2,048 rank vectors × 4 bytes = **16 KiB**, the Golay pattern bytes coming from the
state and branch bits — replaces the 528 per-region tables. For p = 1 its region is exactly the lowest-norm region; for
p = 0 it is a compromise between two coordinate profiles. Measured twice on the F1b harness, same blocks, same process
(*measured*, `llvq-bench/examples/f1rankbench.rs`): **89.05% against 89.69% for exact F1 on 2,000 blocks (−0.64 pp),
88.88% against 89.48% on 4,000 (−0.61 pp)**, paired MSE loss 1.7 ± 0.2%, ball-12 control 92.00% both times, zero blocks
outside Λ₂₄, encoder exact on 99.8% of section targets. The signed prediction before the run was a loss of 2.0 to 4.5 pp.
On the measured curve that table costs 0.34 to 0.66 ms in today's geometry — under H — and the unknown moves to the
arithmetic decode (~260 operations per block, *estimated*, never compiled), which is where E1v died at 79 registers.

Projected on the fundamental criteria for the 4B (*estimated* unless stated; the arithmetic is in the audit reports):
disk identical; VRAM 2.57 → **1.36 GB** and 5.162 → **2.76 b/param** whole model (*computed*); tok/s 100.6 → 82 to 98
with the 67/9 table, ≈ 100 to 108 with the universal table *if* the ALU decode stays under 0.5 ms; ppl and MMLU
unmeasured (F1c is the first gate); admissible class under the triplet 43.3 → 81 to 101 B parameters, the 70B fitting
in 19.5 GB (*computed*) — though the `rot_apply` wall of [format-noyau](format-noyau.md) §8 closes the served path past
the 14B whatever the format. Two blockers the floor never named: the bench encoder runs at 240 ms/block/core against
F1c's 656 µs gate, **366×** (*measured*), so a production encoder precedes any F1c; and F1e's "≤ 2.6 b/param" was
unreachable at 4B by construction with the q8 embedding (2.76) — rewritten on the triplet's b_max.

## 6. Open decisions

- Wave 2 is open, **$2.00 cap** (operator, 2026-09-04), $0.02 spent on the table floor. Content: F1b done at $0; the
  ALU floor of the universal-table decoder at ≤ $0.10 (operator go, 2026-09-05); a production encoder, then F1c on the
  Mac at $0; then F1d at ~$1.00 on L40S, only if F1c passes. Objective set by the operator on 2026-09-05: **an F1 that
  can be tested**. Q5's served run moves to wave 3, after
  F1's verdict: F1c produces a format v2, so sealing a v1 artifact with `v_proj` in int4 now would be building it
  twice. Wave 1's $0.05 overrun stays recorded against wave 1. Project total to date: $97.56 (*measured*,
  `docs/data/jobs.csv`).
- Not in wave 2, and not asked for: F1e (~$8, only if F1c and F1d pass), Q1 at 4B (~$7), the 32B point (~$62).
- A third draw for the attribution, ~$2.14 and a wave-2 cap, operator. Seed 1 (58.02% MMLU) never received the
  eleven arms. Two draws do not make a distribution, and the head of the ranking is what changed.
- Product triplet in force (operator, 2026-08-16): 8k context, 5 GB margin, 32 GiB unit, offload as reference
  only. It leaves 27.93 GB to the weights, so b_max = 3.00 kernel b/weight; `Planes14` exceeds it by 60%. The largest
  admissible class is 43.3 billion parameters at 5.162 b/param (upper bound, embedding 9.7%) and 45.8 billion at
  4.878 (embedding ~2%). The 32B is the served object, the 70B does not fit (*computed*,
  [note-produit-2026-08-13.md](archive/note-produit-2026-08-13.md) §B bis).
- Q1 prereg at 4B, `ots upgrade` of the three timestamps of 09-02 after anchoring: operator.

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
