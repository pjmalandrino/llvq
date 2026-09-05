# Roadmap

What comes next for the project, with its gates and its costs. State as of 2026-09-04. The past is
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
seed. M2 cost ~$2.17 and the replicate $2.14 (*measured*,
[m2rep-graine3-4b-2026-09-04](mesures/m2rep-graine3-4b-2026-09-04.txt)); wave 1 closes at $4.60 of 5.

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
| M2rep | the same eleven arms on the seed-3 artifact of F5 | $2.14 (*measured*) | target holds across the two draws | attribution draw-dependent | done, target retained, ranking not |
| M2b | `v_proj` in int4 g128 dequantized, constant file | ~$0.29 (*computed*, [journal](mesures/m2b-v4bits-2026-09-02.txt)) | G4 ≥ 3.0 and CI > 1.5 | G4 < 1.5 | done, read as cashable 09-04 |
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

M2b is delivered. `v_proj` in int4 g128 gives 59.19% of MMLU, +3.60 pp
[1.47; 5.79], McNemar 2.0e-4, or 80.4% of the f16 gain (*measured*,
[m2b-v4bits-2026-09-02](mesures/m2b-v4bits-2026-09-02.txt)). Memory goes down: 5.149 b/param
(*computed*, same journal). `Planes14` unfolds to 4.804 b/weight (*measured*,
[c1-planesbench-2026-08-06](mesures/c1-planesbench-2026-08-06.txt)). int4 g128 serves the same
content at 4.250 (*computed*, 4 bits plus an f16 scale and bias per group). Serving `v_proj` in f16
would cost +0.263 b/param, that is 5.425 (*computed*, same journal), above AWQ. Line 1 requires a CI
entirely above 1.5; the lower bound is 1.47, and over eight bootstrap seeds it runs from 1.42 to
1.49, never above 1.50 (*measured*, same journal). Lines 2 and 3 require G4 < 3.0
([preregistration-m2b-v4bits-2026-09-02-ECARTS](../proofs/preregistration-m2b-v4bits-2026-09-02-ECARTS.md)).

On 2026-09-04 the operator reads the uncovered case as cashable: both axes move at once, +3.60 pp for
−0.013 b/param, nothing is paid for the gain. Q5 opens. Line 1 stays failed on its letter and the rule
is not repaired after the fact (same file, §É4). What the decision does not settle: M2b dequantizes
`v_proj` to f16 before the matvec, so no kernel serves it in four bits and the served quality of Q5 is
unmeasured; and the +3.60 pp rests on a single quantized file.

The replicate ran on 2026-09-04, $2.14, on the seed-3 artifact of F5 (*measured*,
[m2rep-graine3-4b-2026-09-04](mesures/m2rep-graine3-4b-2026-09-04.txt)). `v_proj` is retained by the
preregistered clause, its CI [+1.11; +4.68] overlapping M2's [+2.39; +6.61], but no difference between
draws is resolved and the head of the ranking swaps: `gate` from 1st to 4th, `down` from 4th to 1st.
The f16 ceiling of `v_proj` falls to +2.87. M2b was then replayed on that same seed the same day, $0.45:
**+2.71 pp [+0.59; +4.93]**, McNemar 0.0106 (*measured*,
[m2b-graine3-4b-2026-09-04](mesures/m2b-graine3-4b-2026-09-04.txt)). The CI clears zero, so the gain is
confirmed on a second draw and Q5 is adopted. The served figure is the range **+2.71 to +3.60 pp**. The
survival rate of the f16 gain into int4 is 94.4% here against 80.4% on the published file, so it is not a
constant and the +2.31 pp extrapolated from it was an artefact. Detail in [`ETAT.md`](ETAT.md) §5 ter and
§5 quater.

New knobs: `LLVQ_RESTORE_F16` and `LLVQ_RESTORE_Q4` (`mmlu`, `ppl`, require `LLVQ_MODEL`),
`LLVQ_H_SHRINK` (`smoke`).

### 2.2 Axis F, format without unfolding

F1 codes Λ₂₄ as a three-section E₈ coset code (Forney 1988, Lepowsky-Meurman 1982). The word is 48
bits: `[state 8][s₁ ~13][s₂ ~13][s₃ ~13][gain 1]`. Decoding costs three lookups and two additions.

| id | lead | cost (*measured* if done, `jobs.csv`; *estimated* otherwise) | adoption | kill | state |
|---|---|---|---|---|---|
| F1a | count states and alphabets for 47 bits, prove the bijection | $0 (*measured*, one session) | **no gate** — a feasibility blocker: it must fit the card's 99 KiB opt-in, tile included | does not fit | states green, bijection **proved**; 92 KiB with the odd-coset restriction, 1,124 KiB without |
| F1b | codebook in `llvq-bench`, 20,000 blocks, 48 bits packed | $0 (*measured*, 64 min Mac) | **no gate** — a measurement: retention against an in-process ball-12 control, feeding F1c's signed prediction | none | **done**: 89.55% against 92.00% (−2.45 pp), 12/16/11 89.38, 13/13/13 85.96 ([journal](mesures/f1b-retention-2026-09-04.txt)) |
| F1 floor | decoder-table floor on L40S, `f1floorbench` | $0.02 (*measured*, three attempts) | **no gate** — a measurement of the lookups alone | none | **done**: D(8 KiB) 0.344, D(16 KiB) 0.663, L2 plateau 4.52 ms; the 67/9 table sits at 2.4–2.5× the F1d budget on the real access distribution; a **universal 16 KiB table** loses 0.6 pp of retention and prices under it ([journal](mesures/f1-plancher-table-2026-09-05.txt), [ECARTS](../proofs/preregistration-f1-plancher-table-2026-09-04-ECARTS.md)) |
| F1 ALU | floor of the universal-table decoder, compiled: word read + arithmetic decode + 16 KiB table, in `nullk`'s process | $0.01 (*measured*) | **no gate** — a measurement informing F1d | none | **done**: decoder verified on the card against the Rust reference (64,512 blocks), 48 registers, 0 local; T = 3.346 ms = 1.20 × B; the excess is arithmetic (~2 ms), not the table ([journal](mesures/f1-rang-plancher-2026-09-05.txt), [ECARTS](../proofs/preregistration-f1-rang-plancher-2026-09-05-ECARTS.md)) |
| F1 ALU v | the same decoder, three arithmetics (no I2F; F₂-algebra patterns; PRMT byte tables), six arms in one process | $0.00 (*measured*, 4 s) | **no gate** — which writing F1d takes | none | **done**: all three equal `tv_f1r` on 30,720 rows, 40 registers; **v3 T = 1.694 ms = 0.61 × B**, v1 1.719, v2 3.444; the int→float conversions were the cost, the read chain was not ([journal](mesures/f1-rang-variantes-2026-09-05.txt), [ECARTS](../proofs/preregistration-f1-rang-variantes-2026-09-05-ECARTS.md)) |
| F1c | format v2, encoder, 0.6B 28 blocks | $0 | **quality + encoding cost**: ppl gate — form to be set by the operator (the ±1 cross-seed range has no power at ρ = 1; proposal: paired Δ per seed at fixed ρ, signed prediction); encoder ≤ 656 µs/block/core, encoder-only figure | out of band on 3 seeds | **first gate of the axis**; encoder prototype at 290–296 µs on Gaussian blocks ([journal](mesures/f1-encodeur-prototype-2026-09-05.txt)), to be measured on real GPTQ residues before the prereg |
| F1d | `tv_l3e8` arm in `planesbench`, QTIP control in the same process | $1 | **throughput + VRAM**: t ≤ t(`Planes14`) measured in the same process, and ≤ 2.20 b/weight kernel | t > t(`Planes14`), i.e. slower for fewer bytes | **to write with the v3 decoder** (v1 equivalent): 0.61 × B on the floor, ≈ 113 tok/s projected at the 4B (*estimated*); F1d measures it with Planes14 in-process, with the gain scale and real labels |
| F1e | 4B sealed in v2, `fusedrun`, paired MMLU | $8 | **the four axes at once**: kernel ≤ 3.00 b/weight (the triplet's b_max; "≤ 2.6 b/param whole model" was unreachable at 4B by construction with the q8 embedding, 2.76), MMLU ≥ 55.59 − 2 SE, tok/s ≥ 100.6, disk ≤ today's | MMLU < 53% | the axis's verdict; the tok/s threshold is the operator's to confirm |
| F2 | sequential trellis + trellis shaping, A3 geometry only | like F1 | fallback if F1a or F1b dies | none | not budgeted |
| F3 | per-row cap, 44 to 50 bits/block, guided by M2 | $7 | +2 pp paired MMLU at constant b/param | < +1 pp | after D3, conditional on F1c |

F1b is under its kill, and since 2026-09-04 the number is exact rather than projected. If the shaping
region is the product of three 8-dimensional balls, the retention is **89.10%** at 2.000 b/dim against a
kill at 90.3% (*computed*, closed form, `ops/f1a_shaping.py`). The definition of retention is checked
against a known value on the way: the served MSE gives back 92.14%, the number of
[fiche-4b](fiche-4b.md). The sphere shaping gain is 0.7292 dB in dimension 8 against
1.0958 dB in dimension 24; the 0.3666 dB lost is +8.81% of MSE on the served 0.077718
([fiche-4b](fiche-4b.md)). This supersedes the bracket of 88.9 to 89.6% (*estimated*,
[projection-gains-2026-09-01](archive/projection-gains-2026-09-01.md) §1.4).

The product is a hypothesis, not the construction: the three sections are chained by the 8 state bits.
What F1a has left to settle is how much that coupling buys back. The region must reach 0.8742 dB of
shaping gain to clear the kill, that is 39.6% of the gap between the product and the 24-dimensional
ball, and 0.9585 dB to be adopted at 91.0%, that is 62.6% of it (*computed*, same closed form). Below
that, F1 stops at its gate.

### 2.2 bis Three gates removed, 2026-09-04

All three F1 gates were found unsound on the same day, each for a different reason. The operator's
standing rule, set the same evening, is that **a gate is written on a fundamental criterion and
never on a proxy** ([METHODE](METHODE.md) §1) — and all three were proxies: Gaussian retention,
table kibibytes, a competitor's milliseconds in the competitor's own grid. So they are not
rewritten on better thresholds; the two that cannot measure a fundamental criterion stop being
gates, and the axis's decisions move to F1c, F1d and F1e where quality, throughput and VRAM are
what is actually measured. What follows records why each fell, because the reasons are the
evidence for the rule. ⚠️ They were audited *after* computing that F1 fails them
as written, which is the shape of a moved goalpost. Two guards against that: no reason below uses
an F1 result as its anchor, and removing a gate is not the same move as loosening one — the F
axis now has **fewer** decision points, all of them later and all of them on measured
fundamentals, so F1 has to survive perplexity at 0.6B and then MMLU, throughput and b/param at 4B
before anything is adopted.

**F1d — it fired on a free decoder.** As written the thresholds were 1.15 and 1.5 times QTIP's
2.246 ms, measured in QTIP's own `<<<128, 1024, 64 KiB>>>` grid, while our launch floor alone is
2.306 ms in ours ([format-noyau](format-noyau.md) §6 forbids that subtraction and hard rule 5
forbids the division). A decoder costing *nothing*, reading the 0.98 GB its own 2.159 b/weight
implies at the 836 GB/s net rate, lands at 2.306 + 1.17 = **3.48 ms** — past the old kill of
3.369 ms. The gate fired on arithmetic that had nothing to do with F1. Restated on our own floor
plus the arm's own traffic, which is the only comparison a single grid supports.

**F1a — 16 KiB was a guess predicated on a factorization that does not hold.** The number came
with the words "after factorizing bases × signs"
([ROADMAP-RECHERCHE](archive/ROADMAP-RECHERCHE.md):130); the truncation study measured that
factorization and the sign action leaves **67 orbits** on the end-section cosets and 9 on the
middle, not one ([f1-regle-de-troncature](archive/f1-regle-de-troncature-2026-09-04.md)). The
card's own attributes are measured and are the honest bound: on L40S,
`MAX_SHARED_MEMORY_PER_BLOCK` 49,152 B, `_OPTIN` 101,376 B, per SM 102,400 B (*measured* at
preflight, [format-noyau](format-noyau.md) §8). The matvec already stages a 12 KiB activation
tile. So 3 × 16 KiB + tile = 60 KiB fits the opt-in; the 80 KiB odd-coset variant + tile = 92 KiB
fits with 7 KiB to spare; the 1,112 KiB variant cannot be in shared at all and would live in L2,
where 48 MB makes capacity a non-issue and latency the question. ⚠️ Fitting is not the same as
being fast: 92 KiB per block against a 100 KiB per-SM budget is **one block per SM**, and A3
measured eight occupancy variants without finding a portable one. The gate is therefore a
feasibility bound, and occupancy moves to F1d where it can be measured.

**F1b — the old gate could not be passed by anything.** "Retention ≥ 91.0%, kill < 90.3%" was
set as "no worse than the shell-12 codebook we buried" ([ROADMAP-RECHERCHE](archive/ROADMAP-RECHERCHE.md):131).
Shell 12 was buried for quantizing worse **at the same cost** — same 48 bits, same VRAM, strictly
dominated ([BACKLOG](archive/BACKLOG.md):130). F1 is not dominated: it halves the VRAM. The gate
imported a domination argument into a case with no domination, and the ceiling makes it
unsatisfiable: any per-section region is a product of three 8-dimensional regions, the best of
those is the ball, so 0.7292 dB is the maximum and the kill needs 0.8742 dB. A criterion no
implementation can meet is a rejection wearing a gate's clothes.

It is not replaced. Retention on a Gaussian source is two transpositions away from quality, and
this repository has measured how loose the second one is: the paper's 4B reads 17.05 ppl for
60.7% MMLU where ours reads 16.94 — better perplexity — for 55.59. **F1b therefore carries a
measurement, not a gate**, and its number feeds the signed prediction F1c is read against. The
first gate of the F axis is F1c, on perplexity.

The served path freezes the gain field at 1 bit: 8 assertions, 4 shaders,
`llvq-cuda/src/planes14_host.rs:113` refuses any other value (*measured*, grep). The v2 format of
F1c, the per-row cap of F3 and any Q arm that changes the code reopen the runtime layout on top of
the quantizer.

### 2.2 ter The floor, and who kills, 2026-09-05

The decoder-table floor ran ($0.02, [journal](mesures/f1-plancher-table-2026-09-05.txt)) and was reported the
same morning as a kill. It was not one: its prereg §1 says the floor decides nothing, and the operator's rule of
the day — **a kill is written on a fundamental criterion and by the operator alone**, [METHODE](METHODE.md) §1 —
puts the verdict elsewhere. The audit that followed kept the number (the 67/9 table sits at 2.4–2.5× the F1d
budget on the real access distribution) and struck the reading (six blocks per SM, L1 28 KB, shared-memory arms
confounded with occupancy and staging; [ECARTS](../proofs/preregistration-f1-plancher-table-2026-09-04-ECARTS.md)).
It also produced the decoder that fits: a universal 16 KiB rank table, −0.6 pp of retention against exact F1,
measured twice. What remains between here and F1c, in order: the compiled floor of that decoder (F1 ALU, ≤ $0.10),
a production encoder against the 656 µs/block gate, then format v2 and the 0.6B run. Objective set by the operator:
**an F1 that can be tested.**

**Evening of 2026-09-05.** The compiled decoder ran: 1.20× B, 48 registers, verified on the card. The
encoder was prototyped at 290–296 µs/block/core (gate 656), the bench's points returned. The format-v2 map is
drawn: `llvq-search` (word map, rank table, trellis, encoder), `llvq-quant` (`F1ShapeGain: BlockQuantizer`),
`llvq-artifact` (header v5 with a second fingerprint, `PUBLISHED_FINGERPRINT` untouched, a disk→word transcoder
because the disk is MSB-first), `llvq-llm` wiring, then a 3-block pilot on the 0.6B — oracle, smoke,
`verify_artifact`, seal, ppl — before the three seeds. Order, each step mutation-tested: (0) the encoder on real
GPTQ residues, 0.5 d; (1) word map and trellis in `llvq-search`, 0.5 d; (2) the production encoder and
`F1ShapeGain`, 1–2 d; (3) format v5, 1–2 d; (4) wiring and the pilot, 1 d; (5) the F1c prereg (gate form: operator)
and the three seeds, 1 d + ~35 min of Mac per seed. *Estimated* 5–7 days.

**Later that evening.** The operator's go — test a faster arithmetic, else return to the measured decoder — ran
three independently written kernels for the same table and word in one six-arm bench: the int→float conversions
were ~1.6 of the ~2.0 ms, the dependent read chain nothing. **v3: T = 1.694 ms = 0.61 × B**, 40 registers; F1d takes
it. The encoder's real-block measurement closed its last assumption (298 µs/block/core on 20,000 rotated GPTQ
residues of the 0.6B, ratio 1.00 to Gaussian). What remains before an F1 the operator can test is the format-v2
work of the plan above, and the form of F1c's gate.

### 2.2 quater Trio: the path to a `.llvq`, planned 2026-09-05

The format's name is **Trio**, proposed to the operator. The three disjoint octads that split the 24
coordinates are what make the 8-bit state, the three sections and the 16 KiB table possible; the name is
short, reads in both languages, and sits beside `Planes14`, `Slot32` and `Golay70`. Tokens: codebook `trio`
in `smoke`, file kind `Trio` in the v5 header, VRAM layout `trio48` for the served kernel, module
`llvq_search::trio`, constant `PUBLISHED_TRIO_FINGERPRINT`.

The objective is a sealed 4B in Trio with its quality measured, at $0 on the Mac. The card numbers (tok/s,
VRAM) come after, at ~$1 and ~$8. Nothing in this table is a gate except step 5's reading; every step is
mutation-tested before it is called green ([METHODE](METHODE.md) §4).

| step | what it produces | files | effort (*estimated*) | check, written before the step |
|---|---|---|---|---|
| 0 | the Trio word map in the dependency-free crate: trellis, rank table (N0 = 1,240), the 12 linear columns, `decode_word`, `encode_word`, field layout | `llvq-search/src/trio/` (from `llvq-bench/src/f1/rank.rs`, the bench copy stays as the independent yardstick) | 0.5 d | agreement with `llvq_bench::f1::rank::decode_word` on 10⁵ words; `encode(decode(w)) == w` on 10⁶; every decoded word in Λ₂₄; mutants: a swapped octad, a shifted rank, a class block exchanged |
| 1 | the production encoder: closed-form membership, lazy trellis join, two adaptive scales with α and the scale pair fixed on the 4,000 training blocks | `llvq-search/src/trio/encoder.rs` (from `llvq-bench/examples/f1enclazy.rs`), `llvq-bench/src/bin/trioencbench.rs` | 1 to 2 d | same points as the bench encoder on 2,000 blocks × 2 scales; retention on the fixed 2,000 blocks ≥ 88.85 (non-regression, not a quality claim); `trioencbench` ≤ 656 µs/block/core, one core, median of 5; kill: over 656 |
| 2 | `TrioShapeGain: BlockQuantizer`: gain from the norm as today, direction by the Trio encoder, reconstruction mirroring the decoder, 48 bits per block | `llvq-quant/src/quantizer.rs` | 0.5 d | codes → reconstruct equals the evaluated weights bit for bit (`g6_artifact` on Trio); `oracle` on every backend before any number |
| 3 | format v5: magic `LVQ5`, a per-file code kind in the header, the v1 fingerprint untouched plus the Trio fingerprint, the disk-to-word transcoder (the disk is MSB-first, the word little-endian), refusals in every runtime transcoder and in the tools that read indices as classes | `llvq-artifact/src/{format,codebook,runtime}.rs`, `llvq-bench/src/bin/{rtbits,classhist,decbench,decfull,decprofile,lswap}.rs` | 1 to 2 d | legacy headers still read; `PUBLISHED_FINGERPRINT` unchanged; raw passthrough byte-identical at v4 and v5; transcoder pinned on 10⁵ words; mutants: kind ignored at read (the round trip must break), gain bit read at bit 0 (must break) |
| 4 | the wiring: `trio` accepted by `smoke`, writer at version 5, `seal`, `ppl`, `mmlu`, `export` read v5; `verify_artifact` bit for bit; a 3-block smoke test on the 0.6B against `leech1c12` | `llvq-llm/src/{calib,sealed,artifact2}.rs`, `bin/{smoke,seal,ppl,mmlu,export}.rs` | 1 d | the 0.6B 3-block run prints the same rate (2.1656 b/weight) on both arms, seals, reopens, and its ppl is finite; the served 4B file still reads 16.9415 at f16 |
| 5 | the 4B in Trio: prereg with a signed prediction, then `smoke 64 2048 12 4096 metal nogs trio 999 rot` on C4, `seal`, `ppl` at f16, `mmlu` on Metal | `proofs/`, `docs/mesures/` | 1 d of work, ~4 h of Mac for the file, ~1 h for ppl and MMLU | the reading is F1e's quality line on the 4B (MMLU ≥ 55.59 − 2 SE, kill under 53%); disk and b/param computed on the sealed bytes; the operator confirms the thresholds before the prereg is stamped |
| 6 | the card: `tv_trio48` from the v3 decoder with the gain scale, in `planesbench` against `Planes14` (F1d, ~$1), then `fusedrun` on the sealed file (F1e, ~$8) | `llvq-cuda/`, `llvq-llm/src/fused*.rs` | 1 week | F1d and F1e as written above |

Steps 0 to 4 are 4 to 6 days; step 5 adds a day and about five hours of Mac. Steps 0 and 1 can run in
parallel with step 3. Step 6 is independent of step 5 once step 3 fixes the file layout. Two inputs are the
operator's: the name, and the quality thresholds of step 5. The 0.6B seeds of F1c are dropped in this plan;
the 4B is the object, and its quality is read with F1e's line.

### 2.3 Axis Q, quality

| id | lead | cost (*measured* if done, `jobs.csv`; *estimated* otherwise) | adoption | kill | state |
|---|---|---|---|---|---|
| Q1 | H shrinkage in production, ρ in [0.5; 0.9] | $0 at 0.6B, $7 at 4B | range ÷ 2 held, median ≤ +range | none | to do, opened by M1 |
| Q2 | asymmetric target and output weighting | $0 then $7 | Δppl ≥ 2 ranges, 3 seeds | < 1 range | to do |
| Q3 | beam GPTQ, K in {2, 4, 8} | $0, 10 h Mac | Δppl ≥ 2 ranges, σ not increased, encoder ≤ K times | < 1 range | to do |
| Q4a | cross-layer equi-norm, VQ version | $0 | Δppl ≥ 2 ranges | < 1 range | to do |
| Q4b | 24×24 maps on the activation side, diagonal first | $0 then $7 | diagonal Δppl ≥ 3%; full +2 pp | none | to do, full after Q6c |
| Q5 | mixed precision on `v_proj` | $7 and a kernel | ≥ +3 pp paired for ≤ +0.10 b/weight | < +1.5 pp | adopted 09-04, +2.71 to +3.60 pp on two draws, kernel started |
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
| Q5 after the replicate: measure M2b on seed 3, restrict Q5 to the published file, or park it | before any Q5 kernel | nothing, Q5 does not open |
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
