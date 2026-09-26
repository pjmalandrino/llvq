# Quality leads to cross 60 MMLU, survey of 2026-09-12

The served 4B reads **56.37 MMLU micro on the full 14,042-question split** (*measured*, dense
arithmetic, [f1e-census-2026-09-11](mesures/f1e-census-2026-09-11.txt)); the served kernel reads about
56.5 (*computed* from the −0.14 pp paired delta). The target is 60, so **+3.63 pp**. No single lead in
this survey reaches it, and the best stack the repository can build in a month lands at 59.0 to 59.7 on
the served draw. The one path that is not a draw lottery is post-quantization fine-tuning with frozen
codes, and it moves the comparison to the paper's fine-tuned column. This document ranks 38 leads by their corrected central gain, with the trade-off on the
fundamental criteria and the implementation cost. It does not replace [ROADMAP-QUALITY](ROADMAP-QUALITY.md),
which the operator sanctioned on 2026-09-06; it adds rows and corrects ten of that table's cells.

## How the table was built

Four readers went through the repository (the ledger of every lead ever listed, the encoder code
against the paper's recipe, the evaluation and serving path, the paper's own ablations). Six literature
sweeps covered calibration and Hessians, rotations, post-quantization fine-tuning, mixed precision,
the perplexity/MMLU dissociation, and every arXiv posting from 2026-05-01 to 2026-09-11. A synthesis
merged 60 raw leads into 32. Two adversarial verifiers then attacked each lead: one against the
repository record (already measured, killed on a proxy, wrong code citation), one against the
fundamental criteria and the gain arithmetic on the 4B accounting. A completeness critic closed the pass:
it added six rows, settled the verdict disagreements and recomputed the stacks. 76 agents, 1,480 tool
calls, no run, no edit. The gain column carries the corrected central of the two verdicts, with the union of their
intervals.

Accounting used throughout: 3,633,315,840 projection weights, 4,022,468,096 parameters, served
2.8138 b/param whole model and 2.2044 kernel b/weight, margin **0.7956 kernel b/weight** before the
triplet's b_max of 3.00. A move to int4 g128 costs 2.10 kernel b/weight per weight moved, that is
+0.021 kernel and +0.019 b/param per 1% of projection weights (*computed*). Noise: 0.43 pp at constant
file, 2.92 pp for any re-encoding arm; gains compose at 0.62 to 0.79 (*measured*, seven arms).

## The table

Gain in MMLU points on the served base. Labels: *measured* (here), *estimated* (transposed from a
restore arm, a perplexity figure or another model), *reported* (the paper's own number). "Chain" says
where the lead acts. The last column is the cheapest experiment that decides the lead.

| # | Lead | Chain | Gain, pp | Trade-off on the fundamental criteria | Complexity | Status in the repo | Decides it for the least |
|---|---|---|---|---|---|---|---|
| L28 | Fine-tuning ladder with frozen codes: norms, temperature and row multipliers, then the paper's input-column scales, then the f16 tail, trained in torch on the exported checkpoint | fine-tuning | **+1.5 to +1.8** [0; +3.5], *reported* +2.1 by the paper on Qwen3-4B, *estimated* here | Column scales +0.0038 b/param, +0.0042 kernel; masks 1 and 3 cost 0 bits and 0 throughput. The comparison moves to the paper's fine-tuned arms (QTIP-FT 59.5) and the perplexity line stops being comparable to f16 | 7 to 12 days. `export.rs` refuses `Int4G128` and un-rotates, so masks 2 and 3 need a rotated-basis export or a bit-exact torch rotation. Backward pass in torch, not in the repo. $1 to $5 per arm | Never tried; rows 14 and 17 merged; prescribed in `archive/audit-recherche-2026-09-01.md` | Mask 1 (norms, row multipliers, temperature) on 5 M tokens of C4, ~40 min of L40S, scored on the full split at the 0.43 pp bar |
| L24 | Low-rank residual correction: EoRA in closed form, RILQ trained | post-quant correction, bits | **+1.5** [+0.3; +2.5], *estimated* (RILQ +8.1 CSQA on QuIP# Llama-3-8B, trained) | r = 16 f16: +0.131 b/param (2.945), +0.146 kernel (2.35, 18% of the margin), +66 MB. Throughput −16% unfused (+504 launches), −5 to −10% fused. Row 20's "+0.113 at r = 32" is wrong: 0.263 f16 | EoRA 3 to 4 days plus one Hessian capture (none is retained on disk); kernel GEMV path 4 to 6 days; RILQ needs the L28 trainer | Never tried; row 20 | $0: share of the H-weighted error energy in the top 16 directions per matrix; under ~10% the closed form cannot buy a point |
| L04 | Calibration text composition: MMLU-format prompts from the validation split mixed with C4, coverage-based selection | calibration data | **+1.0 to +1.5** [−1; +3.5], *estimated* (TACQ's MMLU figure is same-subset and from a collapsed base; DCLM-edu here read −0.55) | 0 bits. Two Mac encodings per comparison (arm plus a same-device witness), 2.92 pp per re-encode. Calibrating on the dev split is contamination: the harness draws its shots there | 1 to 2 days: a prompt renderer in `corpus.rs` reusing `mmlu.rs` `block()`, one more `LLVQ_CALIB` value | Never tried as a format arm; row 6 | One 50/50 mix at ×1 on the Mac paired with a same-day C4 witness, MMLU on Metal, two draws |
| L03 | Calibration volume ×8 to ×96 with a streamed hidden-state capture | calibration data | **+0.8 to +1.4** [−0.5; +3], *measured* once at ×32 on a naked card base (+2.98 [+0.15; +5.81]); the direct served-base reading V32+Q5 − Tetra+Q5 is +0.22 | 0 bits. Encoding cost ×2.4 at ×32 ($13.60 on rtx-pro-6000x2) to ×5 at ×96 (~$30). The host store must stay f32 to keep the arithmetic: 42.9 GB at ×32, so the Mac takes ×8 today and ×32 only from NVMe | 1 to 2 days: a `HiddenStore` replacing the resident slice at `calib.rs:575/617` | Measured once; `ETAT` §7 and `ROADMAP` :44 still close it on the 0.6B proxy; the journals `jobs.csv` cites do not exist | $0: the streamed capture reproduces the resident perplexity at 131k; then ×8 on the Mac paired with a same-device witness |
| L01 | Ship the radial row-scale constant ρ = 0.929 on the served file | post-quant correction, 0 bits | **+1.0** [−0.3; +2.5], *measured* on bare Tetra (+1.65 [−0.18; +3.54]), *computed* on the served base | None: 0 bits, same kernel, no re-encode, 0.43 pp bar. Perplexity +9.5%, the fourth dissociation | 0.5 to 1 day: `rhoapply` must pass `Int4G128` records through and never scale the tail nor `v_proj` | Measured, not served; absent from the roadmap (M1 postdates it) | One full-split arm, $0.70 on L40S or ~50 min on the Mac; idempotence at ρ = 1 first |
| L30 | Cyclic option marginalisation, four-rotation average | evaluation protocol | **+0.6 to +0.7** [−0.5; +2], *estimated* (label-free offset proxy +0.5 to +0.8 *computed* on the served dumps, negative on the two other Tetra encodings; f16 moves too) | 0 at serving, ×4 at evaluation. A protocol change: `mmlu.rs:26-39` pins the single-order argmax as the comparability contract, so a debiased score is reported beside 56.37, never in its place. The served file over-picks A by 255 of 2,280 (11.2%); row F's counts belong to `Planes14` | 0.5 to 1 day in `mmlu.rs`, a dump v2 with 16 logits; $0 on the Mac or $2.80 for a ×4 full split | Never run; row F | The 2,280 sample cannot resolve +0.7 (paired SE 0.6); one ×4 dense arm on the full split, $2.80, f16 rescored the same way |
| L21 | Next int4 g128 steps inside the margin, chosen by a per-(type × layer) map: `k_proj` all layers, `down_proj` first five layers. Not T3 | mixed precision | **+0.7 to +0.8** [0; +1.8], *estimated* from measured restore arms (k f16 +2.09 / +1.11; T3 − T2 = +1.52 for q+k+o) | k: +0.0545 kernel, +0.0493 b/param, +25 MB (7% of the margin); down first five: +0.072 / +0.065 / +33 MB (9%); both: kernel 2.331, 2.929 b/param. One int4 launch per upgraded matrix. `down_proj`'s rank flips between draws | 0.5 day for a layer band on `LLVQ_RESTORE_*`, then 6 to 21 restore arms (~1 h each on the Mac); 3 to 5 days to key the writer per (type, layer) | Never tried per layer; ancestor P24 | `LLVQ_RESTORE_Q4=k_proj` on the served file: exists, never run, ~1 h Mac, 0.43 pp bar |
| L06 | Intra-block sequential refresh: capture the `o_proj` and `down_proj` Hessians on already-quantized sublayers | calibration data | **+0.5 to +0.7** [−0.5; +1.5], *estimated* (no MMLU figure for the refresh alone; Qronos family +0.6 at 2 bits) | 0 bits. Encoding 2 h 27 to ~3 h 15 on the Mac. 35% of the weights are calibrated today on inputs the served model never sees | 1 to 2 days: split the block loop `calib.rs:705-960` into four capture, factor, quantize groups | Never tried; row C | One encode with the refresh paired against a same-day witness; run it inside the L05 seed set |
| L34 | `v_proj` tier ladder inside the int4 kind: int8 g64 on `tv_q8_h`'s arithmetic, or int4 g32 | mixed precision | **+0.5** [0; +1.15], *estimated*, bounded by the measured f16-minus-int4 gap on `v_proj` (0.16 to 1.15 over three draws); int4 g32 about half | int8 g64: +0.110 kernel (14% of the margin), +0.100 b/param; int4 g32: +0.0195 kernel, +0.0176 b/param, +8.9 MB. Replaces Q5's record, so no further sub-additivity with it | 2 hours for a bits/group knob on the `RESTORE_Q4` path; the reader refuses bits ≠ 4 and group ≠ 128 (`format.rs:1230-1239`) and `tv_q4_h` hard-codes 128 | Never tried at any width but 4/128; new | $0.70: `LLVQ_RESTORE_Q8` g64 on the served file, full split, paired against the census dump; under +0.4 pp the tier axis closes at int4 |
| L12 | Free-magnitude arm, true Spherical GPTQ (`leech1c12f`) measured in MMLU at the 4B | encoder objective | **+0.3** [−1.5; +2], *estimated* (paper Table 9, mean +1.25 over four rotated pairs on Llama-2-7B). The 4B arm confounds 63 against 48 bits a block with the retraction | The test arm is 63 bits a block and not servable. Stored per-block scales: f16 +0.584 b/param and +0.667 kernel (83% of the margin), 8-bit +0.292 / +0.333, 4-bit +0.146 / +0.167 | 1 to 3 days of plumbing (`calib.rs:651-664` refuses an artifact for a free-magnitude codebook; `mmlu` needs a sealed file) plus 2 × 2 h 27 Mac. A stored 8-bit variant is 2 to 4 weeks | Never in MMLU. Design C was killed on a perplexity proxy and implemented Algorithm 3 plus a snap the paper does not have; the `group_scales` kill was on the no-rotation arm | A 0.6B, 28-block perplexity ladder with three arms: coded 1 bit, free 63 bits, free with 8-bit stored Algorithm 3 scales. Only the third is servable, at 2.53 kernel b/weight, and no 4B spend is justified before it |
| L05 | Hessian regularisation: shrink ρ at the 4B, sink-row masking, joint damping sweep | Hessian | **+0.4 to +0.5** [−1; +1.5], *estimated* (perplexity only: −31% and range ÷6.7 at 0.6B) | 0 bits. 7 to 15 h Mac for two ρ × three seeds. The one lead that can lower the 2.92 pp floor | 0 code for the shrink (`LLVQ_H_SHRINK` shipped), 0.5 to 1 day for the row filter in `Hessian::accumulate` | Never run at 4B; rows 5 and E | $0, minutes: share of tr(H) carried by positions 0..3 and by the top 1% rows on 8 windows |
| L08 | First-order error term in the compensation (FOEM): pull the running weights back toward the original during the sweep | encoder objective | **+0.4 to +0.5** [−0.5; +1.5], *estimated* (*reported* +2.3 MMLU at W3 scalar on Llama-3-8B, from a more collapsed base) | 0 bits, encoding unchanged, one new hyper-parameter β | 2 to 4 days: derive the 24-block form, ~80 lines in `gptq.rs:235-330`, mutant test | Never tried; new | 0.6B, 28 blocks, β in {0.05, 0.1, 0.2}, three seeds, read on the range |
| L07 | Compensation-aware target: GPTAQ, Qronos, CAE | encoder objective | **+0.35 to +0.5** [−0.7; +2], *estimated* (the gain decays from −56% to −7.5% of perplexity as the base excess falls; CAE +6.6 pp at 45.6 falls to +0.2 at 65) | 0 bits. Encoding +10 to 40%; a second resident hidden chain halves the volume under the VRAM cap unless L03 streams first | 5 to 8 days in `calib.rs`, `gptq.rs`, `linalg.rs` | Never tried; row 11 | L06 first; then one GPTAQ encode paired against the L06 witness |
| L15 | Zero-bit preprocessing before the rotation: MagR, absorbable column scales (D2Quant DSQ) | encoder objective | **+0.3 to +0.4** [−0.7; +1], *estimated* (DSQ +3.5 MMLU on an un-rotated scalar Qwen3-8B; MagR perplexity only) | 0 bits, folded. Encoding +10 min to +30% | MagR 3 to 5 days; DSQ 2 to 4 days; `oracle` checks the fold | Never tried; row 10 | $0 on the `f1recdump` blocks: rotated block statistics before and after; if kurtosis stays 3.01, close |
| L11 | Hessian-aware gain and candidate selection at the block decision (Schur-conditional metric S_B) | encoder objective | **+0.3** [−1; +1.5], *estimated* (M0 and M1b bound the identity-metric per-block gain near 0) | 0 bits, same words, same fingerprint. Encoding +a few % | 2 to 4 days in `quantizer.rs` and `gptq.rs:269-315`, with an anisotropic toy test | Never tried; [hypothese-metrique](hypothese-metrique-tetra-2026-09-08.md) | $0: fraction of `f1recdump` blocks where the S_B ranking differs from the Euclidean one |
| L13 | The 48-bit gain-bit ladder (`leech0c13`, `leech2c11`, `leech4c10`) replayed under the right correction | codebook | **0** [−2; +2], *measured* (`leech0c13` +1.19 [−1.54; +3.98], Euclidean regime, unresolved) | Not servable: every served layout refuses `gain_bits ≠ 1` (`planes14_host.rs:113`, `runtime.rs:424`) and a Ball codebook unfolds to 4.804 kernel b/weight, 60% over b_max. A new 48-label-bit word is a new point set, decoder and fingerprint: months | 0 days to replay, 3 h 55 Mac and $0.33 a rung; months to serve | Measured, suspended; row 3, whose Table 10 figures are not in the paper notes | Nothing: the family is closed until a servable word exists |
| L17 | Rotation variants: Paley odd factor, two-pass RHT, block-aligned permutation; rotation-seed variance as an instrument | encoder objective | **+0.2 to +0.3** [−0.6; +2], *estimated* | 0 bits for the first two; +0.0022 b/param for a permutation table, −0.5 to −1% tok/s for its gather | 1 to 2 days, 1 day, 3 to 5 days; seed variance 0.5 day plus two encodes | Never tried. Row 9's refutation rests on calibration seeds; the rotation seed was never varied | $0 on the `f1recdump` blocks: 99th-percentile block kurtosis before and after |
| L20 | Q5 served: pair the mixed file, and route `v_proj` through GPTQ instead of skipping it | mixed precision | **+0.2 to +0.3** [−0.5; +1.15], Q5 itself is inside the 56.37; the GPTQ variant is *estimated* | 0 incremental. Encoding +~4 min. The mixed file's perplexity moved the wrong way (×1.334 against ×1.3203) | Pairing 0 days; the variant 2 to 3 days in `calib.rs:793-812` | Served; row 1 | $1.40: full-split arms for bare Tetra and f16, so 56.37 pairs inside the repository |
| L23 | Super-weight rows of `down_proj` in f16 (early layers); sink-direction rank-1 correction on `v_proj` and `o_proj` | post-quant correction, bits | **+0.2 to +0.3** [0; +1.5], *estimated*, bimodal | +0.0004 b/param; −0.5 to −1.5% tok/s from extra launches | 1 day of diagnostics; 2 to 4 days for a per-row restore knob and a side-list record | Never tried; new. Row E is the input-side cousin | $0, one hour: one f16 forward for the per-layer maxima and the served reconstruction error along the sink direction |
| L25 | Per-output-row bias b = E[(W − Ŵ) x], closed form, post hoc | post-quant correction, bits | **+0.2 to +0.3** [−0.3; +1], *estimated* (DAC +0.33 MMLU on a scalar 2-bit Qwen3-8B) | +0.0044 b/param, +0.0049 kernel for an f16 field; one add per row | The eval arm costs nothing; 2 to 3 days for the field and the epilogue if it resolves | Never tried; new | $0 on the Mac: compute b on the candle side, apply it in a `RESTORE`-style arm, 0.43 pp bar |
| L14 | Block-of-24 sweep order: act-order at block granularity, min-pivot | encoder objective | **+0.1 to +0.2** [−0.3; +0.8], *estimated* (every act-order gain comes from un-rotated outlier models) | 0. Pivoting changes the factor: extend `both_factorizations_agree` | 1 day, `gptq.rs:255`, `linalg.rs:68` | Never tried; row D | $0: residual spread of diag(Q'HQ) per 24-block on the encoded artifact, shared with rows 13 and E |
| L33 | MSE-optimal clipping for the int4 `v_proj` groups in place of min/max RTN | mixed precision | **+0.15** [−0.1; +0.4], *estimated*, bounded by the same f16-minus-int4 gap | 0 bits, same bytes, same kernel; minutes of encoding; constant-file arm | 0.5 day: a clip-search sibling of `quantize_affine` in `embedquant.rs`; the pinned instrument `quantize_dequantize_q4` must not change | Never tried; new | $0: weight-space MSE at α = 1 against the best α per group on the 36 matrices, then the H-weighted error from L36 |
| L02 | Per-row scale refit in the Hessian metric, Algorithm 3 collapsed to one scalar a row | encoder objective | **0 to +0.3** [−1.3; +1], *estimated* (the identity-metric form equals the constant to 0.04 pp, M1b; the H metric may pull the scales back toward 1 and undo part of L01) | 0 bits. Post-hoc form: one capture pass with served weights, 20 to 40 min Mac; interleaved form ×K encodes. Counts once with L01 and L28's mask 1: one degree of freedom | 1 to 2 days beside `refine_group_scales` | Never tried in the H metric; new | $0 from L36: the distribution of the closed-form s_i. Near 1.0 with small spread closes the row axis on the constant; near 0.93 with structure earns one $0.70 arm |
| L09 | Teacher-decision token weighting of H (SchurQuant Eq. 19), forward only | Hessian | **0** [−1; +1], *estimated* (the paper reports a six-task zero-shot mean, no MMLU, and no isolation of the weighting; Eq. 19 carries a 1/r_i term the lead had dropped) | 0 bits; +36 forwards over the calibration set, 1 to 3 h Mac. Exclusive with L05's sink mask and L10: one weighting is served | 2 to 3 days | Never tried; new | $0: served-versus-f16 top-1 flip rate at prefix depths 9, 18, 27, 36 on 8 windows; already ≥ 14.7% at full depth, so only the depth profile is open |
| L19 | Re-qualify the KeepExact tail by salience | format | **+0.1** [0; +0.5], *estimated* | +0.0022 b/param for a permutation table; breaks the fingerprint | Days to weeks | Never tried; row 13 | The same $0 diag(Q'HQ) spread as L14 |
| L35 | Row scales at m = 2 to 4 segments a row, the middle of the Algorithm 3 ladder | post-quant correction, bits | **+0.1** [−0.5; +0.6], *estimated* (per-row freedom bought −0.04 over one constant; after the Hadamard the segments of a row are exchangeable to first order) | +0.0043 b/param and +0.0047 kernel per extra f16 scale a row (m = 4: +0.013 and +0.014, 1.8% of the margin); a segment multiply in the kernel | 0.5 day diagnostic; 2 to 4 days to serve | Never tried at any m between 1 and d_in/24; the M1 journal names it the column axis of chantier 14 | $0, half a day: fit m = 2 identity-metric scales on the served file, tail excluded; if 95% of rows sit within ±2% of 1, close |
| L27 | Sparse f16 outliers by weight salience (SqueezeLLM, SpQR) | mixed precision | **+0.1** [0; +0.4], *estimated* (perplexity only, 3 bits, scalar) | +0.144 kernel (18% of the margin), +0.130 b/param, a third launch; a worse rate than the int4 tier | 1 to 2 weeks, encoder and kernel | Never tried; row 18. A sparse overlay was measured as a kernel arm (`Planes12x`) | $0: share of Tr(ΔW H ΔWᵀ) in the top 0.45% weights |
| L10 | Output-gradient-weighted Hessian (GuidedQuant, KronQ, REAL-Q) through a one-off torch gradient export | Hessian | **0** [−0.5; +1], *estimated* (perplexity only; QTIP −10.4%, shrinking to −1.8% at 70B) | 0 bits; one torch backward over the calibration set; factor time × g | 5 to 10 days; backward in torch | Never tried; row 19 | Gradient traces per output row on 8 windows in torch; if rows differ by less than ×2, close at $0 |
| L16 | Search harder on the local proxy: K-best beam, combinatorial ADMM | encoder search | **0** [−1; +1.5], indeterminate | Encoding ×K (~10 h Mac at K = 4) | 2 to 3 weeks | Row 12; three precedents where a better local proxy composed worse | L11 first |
| L26 | Embedding and tied head in int4 g64, a financing lever | mixed precision | **−0.35** [−1.2; +0.5], *measured* on a `Planes14` base, under the bar | −0.40 b/param, kernel 0, disk −559 MB; a second decode kernel for the head | 1 to 2 days to serve | Measured; row 4 | Only if L21 needs the whole-model figure financed |
| L29 | Output temperature, one scalar in the final RMSNorm | serving numerics | **0**, *computed* (argmax invariance) | None | 0.5 day | Row B | One Metal perplexity run; it reframes the perplexity gap, never MMLU |
| L22 | Protocol settlement: micro against macro (58.42), the f16 tie rule, full-split replays of f16 and `Planes14` | evaluation protocol | **0** [−0.13; +2.05], *computed*: the target moves by 2.05 only if the paper reports the unweighted mean; the repository adopted micro on 2026-08-01 | None. `Iterator::max_by` returns the last maximum, so exact f16 ties go to D: 0.16 pp on the kernel census dump, and the whole −0.14 kernel-versus-dense delta | 0.5 day; $1.86 for the two replays | Items new; row A done | $0: `mmlupair --intersect` of the census dump against the 19 flat dumps |
| L31 | Instruments: M3 (STEM column, attention entropy), flips and KL to f16, recall-std, per-layer sensitivity map | instrument | **0** by construction | None | 0.5 day for the dump statistics, 1 to 2 days for the rest | Not built; M3 section | $0, half a day: the `mmlupair` columns on the existing dumps |
| L36 | Capture-only pass with the served weights and dense H retained | instrument | **0** by construction; it decides ten rows (L02, L05, L06, L14, L15, L19, L23, L24, L25, L27) | None. One Mac hour once the code exists: pass 1 took 394.9 s on the 4B, `down_proj`'s H is 378 MB | 1 to 2 days: a capture-only mode in `calib.rs` on `artifact2::load` weights, dense H kept per activation, a stats binary in `llvq-bench` | Does not exist: the artifact stores no H, `f1recdump` keeps none, `calib.rs` drops dense H after factoring | It is the cheapest decisive measurement of the table |
| L37 | Record transplant into the served v5 file: every mixed-precision or post-hoc step as a constant-file arm | instrument | **0**; it moves the bar of L21, L26, L33, L34 and L35 from 2.92 pp to 0.43 (0 on the full split) and removes 2 h 27 of Mac per arm | None on the object | 1 day: a record walker over `read_record` and `write_record` with kind substitution; `rhoapply` and `embedq` refuse kinded files today | Partial | $0: idempotence sha256 and bit-for-bit agreement with the restore path |
| L38 | Two more encodings of the served recipe at `LLVQ_CALIB_SEED` 1 and 2, scored on the full split | measurement design | **0**: the recipe's mean and sigma, published as a distribution, never as a best draw | None. ~4 h Mac and $1.40; a third seed +2 h and $0.70 | 0 days of code; operator go and a stamped prereg | Never run: seeded draws exist only for `Planes14` on the card | The encodings themselves |
| L32 | Encoder numerics and the noise floor: the device confound (−1.18 pp on one pair), f64 accumulation, dead-column guard, draw counts | measurement design | **0**, a design fact | Sets the rule: three draws per re-encoding arm, or measure the variance reduction first | 0.5 day | New as a row | $0: encode block 0 twice with f32 against compensated accumulation and count differing indices |
| L18 | Witness re-encode of `leech1c12` | control | **0** | 4 h Mac. Makes the 2.92 pp readable as a pure re-encode delta | 0 days | Declined 2026-09-06; row 2 | The re-encode itself; it is also L12's witness |

## What crosses 60, on the corrected centrals

Counting rule first. The row-scale family (L01, L02, L28's mask 1, L11's gain arm) is one degree of
freedom and counts once, at +1.0 to +1.5. The three token reweightings of H (L05's sink mask, L09, L10)
are exclusive. L06, L07 and L08 are one drift-correction family. L23 and L25 are the same
constant-direction shift. L21, L24, L27 and L34 all draw on the same 0.80 of margin. Composition uses
the measured sub-additivity, 0.62 to 0.79; the one cross-family pair measured, Q5 with the volume,
composed worse, at 0.562.

| stack | leads | sum of centrals | after sub-additivity | expected | cost on the criteria |
|---|---|---|---|---|---|
| constant file, zero bits, no re-encode | L01 + L25 + L33 | 1.35 | +0.8 to +1.1 | **57.2 to 57.5** | none; no noise on the full split |
| bits inside the margin, constant file through L37 | plus L34 int8 `v_proj`, L21 (k and the down band), L24 (closed form +0.5, trained +1.5) | 3.15 to 4.15 | +2.0 to +3.3 | **58.4 to 59.7** | kernel 2.59, 48% of the margin; 3.16 b/param whole model, 0.53 behind QTIP at equal embedding; tok/s −5 to −10% fused, −16 to −32% unfused; the trained forms need the torch trainer |
| plus the re-encoding leads | plus L03 ×32, L06, L05, with L28's mask 1 in place of L01 | +1.0 to +1.5 more | | **59.0 to 59.7**, interval [57.5; 61.5] | the base becomes a fresh draw at 2.92 pp; the calibration leads share one estimator and compose worse than 0.62 among themselves |

The honest expected value of the best stack the repository can build in a month is 59.0 to 59.7 on the
served draw. The point estimate does not cross 60. What crosses 60 is the draw: a fresh encoding of the
served recipe, unchanged, passes 60 with a probability near 11% if its mean is 56.4 and its sigma 2.92
(*computed*, P(z > 1.24)); the expected maximum of three draws is the mean plus 2.5 pp, of eight draws
plus 4.2 pp. Row 9's rule holds: the best of N draws is a selection artefact, not a gain. The paper's
60.7 is itself one draw at an unstated sigma, so the fair comparison is recipe mean against recipe
mean, and the served recipe's mean is unknown until L38 runs. The one path to 60 that is not a lottery
is the fine-tuning ladder, L28, and it moves the comparison to the fine-tuned column of Table 6
(LLVQ-FT 62.8, QTIP-FT 59.5). Two reservations travel with every figure above: the sub-additivity was
measured on f16 restore arms, which overlap by construction; and the served recipe's deficit doubles from
the 4B to the 8B, so a 4B stack tuned to 60 says nothing about the classes the triplet targets.

## The measurements that decide the most rows for the least

1. **Already done on the dumps, $0**, and to absorb into the roadmap: the served file pairs +2.03
   against bare Tetra and −0.07 against `Planes14`, both unresolved (L20); the tie rule is worth −0.13 pp
   on the kernel dump (L22); the label-free bias proxy is +0.5 to +0.8 on the served file and negative
   on the two other Tetra encodings (L30); flips against f16 run 34 to 37% against 11% for AWQ (L31).
2. **L36**, one capture pass with the served weights and dense H retained: 1 to 2 days of code, one Mac
   hour, ten rows decided, most of them to zero. Nothing else in the table comes close.
3. **L01 on the served file**: ρ recomputed on the 216 lattice records, the idempotence control, one
   full-split arm at $0.70.
4. **L37**, the record transplant, one day: L21, L33, L34 and L35 become constant-file arms at $0.70
   each. The 2,280 sample resolves none of them (paired half-width 0.85 to 1.7 pp); the full split does
   (paired SE about 0.17 pp).
5. **L38**, two seeded encodings of the served recipe, ~4 h Mac and $1.40: the recipe's mean and sigma
   before any re-encoding lead is read.
6. The re-encoding leads, in one file: L05 at three seeds read on the perplexity range, L03 at ×8 on the
   Mac with L06 folded in, then L04.
7. **L28's mask 1**, about $1.2 of L40S, once `export` handles `Int4G128`.

L12, L13 and the margin leads wait for those results.

## Corrections the sweep forces on the living documents

- `ROADMAP-QUALITY` row 20 prices r = 32 at +0.113 b/param; the shapes give **0.263** in f16, which
  `archive/ROADMAP-RECHERCHE.md` :186 already carried.
- Row 4's −0.4049 b/param does not reproduce: **−0.3868** (8.5 to 4.5 b/weight on 388,956,160 embedding
  weights, confirmed by fiche-4b's e8 2.7961 against e4 2.4093).
- Row 9's refutation was measured on **calibration** seeds (`LLVQ_CALIB_SEED`); the rotation seed has
  never been varied. The verdict stands, the label does not.
- Row 2's premise is wrong: the calibration shard moved from 00000 to 00001 on 2026-08-01
  (`corpus.rs:405-411`), so a witness re-encode is a fresh draw against the published file and pairs
  cleanly only with the encodings of 2026-09-06 and 2026-09-09.
- Row 3 cites Table 10 figures (25.1 against 27.7) that are not in `llvq-paper-notes.md`.
- Row 6's "0 to +2 estimated" is stale against the measured +2.98 of the ×32 arm.
- Row 8's kurtosis 3.01 was measured on Qwen3-0.6B block 0 under a pure Hadamard (k = 1); it bounds
  nothing about the 4B's k = 5 and k = 19 Kronecker factors.
- Row 19's "retraction proof to redo" is stale: Eq. 17 is a no-op under a coded gain (`ETAT` §4).
- Row B's slope 0.4660 belongs to `Planes14`; the served file reads 0.4982, of which only the sd ratio
  (0.67) is temperature; the rest is correlation (0.75).
- Row F's bias counts belong to `Planes14`; the served file over-picks A by 255 of 2,280.
- `ETAT` §7 and `ROADMAP` :44 still close the calibration volume on the 0.6B three-block proxy. The ×32
  arm of 2026-09-07 reads +2.98 pp [+0.15; +5.81] on the same card (`jobs.csv` row 121). `HISTORIQUE`
  stops at 2026-09-07 and has no entry for that run; the two journals `jobs.csv` cites,
  `volume-2026-09-07.txt` and `q5-sur-v32-2026-09-07.txt`, do not exist. `ROADMAP` :268 says ×32 fits
  on no rentable card under 96 GB; the deviation file's own table gives 59.5 GB and the run happened.
- The operator's refusal of T3 on 2026-09-11 is written in no file; only option B of 2026-09-06 is.
- `mmlu.rs` :804-807 breaks exact ties toward the last option, undocumented; −0.13 pp on the kernel
  census dump under first-max, and the whole −0.14 kernel-versus-dense delta.
- The served mixed file was never paired against bare Tetra on the same plan before this sweep; the
  56.95 belongs to a restore arm on another file, and the mixed file's perplexity is ×1.334 against
  Tetra's ×1.3203.
- The reader accepts any nonzero rotation flag, so a file encoded with a rotation variant (L17) would
  decode silently to garbage: a v6 construction enum precedes any variant.

## Dropped, and why

T3 (whole attention int4, refused by the operator, 64% of the margin). Design C (killed on a perplexity
proxy, but subsumed by L12). Stored `group_scales` (+0.584 b/param, 83% of the margin). Best-of-N seeds
(a selection artefact at 2.92 pp). Q4a equinorm (structurally 0). Output rotation (mean 0 in the paper's
Table 9). Dense learned rotations (Cayley SGD, and the Gaussian target is already met at kurtosis 3.01).
Block-AP, PV-tuning, LC-QAT (×100 to ×1000 over budget; their gain credits parameters Tetra lacks).
AWQ 1% f16 (+0.32 b/weight). ICQuant (+0.27 b/param). Channel permutation (format break). Entropy
coding (index entropy 46.65 bits for 47 paid). Shell cap L ≤ 4 (−0.66 pp). KV q8 and the f16 head
(memory levers, quality null). Per-letter additive bias fix (cross-validated −0.51). Inference-time
recovery with an f16 verifier (8 GB against a 5 GB margin). Per-row or per-matrix variable rate (needs
variable-length words). Two-sided Kronecker sweeps (deferred: atomic 24-d words need a new ordering
proof). A standalone damping sweep (null at 0.6B, folded into L05). Group-scale training (row 15, the
object does not exist in Tetra). Model scale as the lever (the 4B is the perimeter).

## Provenance

Primary sources read in full or at table level, by lens. Calibration and Hessians: GPTAQ 2504.02692,
Qronos 2505.11695, GuidedQuant 2505.07004, SchurQuant 2608.15567, FOEM 2507.11017, QEP 2504.09629,
KronQ 2607.07964, REAL-Q 2609.00049, DASH-Q 2604.13806, Babai/GPTQ 2507.18553, Provable PTQ 2508.04853,
Self-calibration 2410.17170, COLA 2510.10618, COVERCAL 2604.24008, TACQ 2504.07389, PiSO 2606.10890,
Bielik-Q2-Sharp 2603.04162, the GPTQ reference README. Rotations: HARP 2605.29843, ButterflyQuant
2509.09679, OptR 2608.02691, KurTail 2503.01483, super weights 2411.07191, D2Quant 2602.02546,
SpinQuant 2405.16406, Qwen3 quantization study 2505.02214, Kashin-DCT 2609.11687, QAM-W 2605.26339.
Fine-tuning: LLVQ 2603.11021 (HTML, fine-tuning paragraph), EoRA 2410.21271, RILQ 2412.01129, LQER
2402.02446, EfficientQAT 2407.11062, PV-Tuning 2405.14852, QuIP# 2402.04396, QTIP 2406.11235, Norm
Tweaking 2309.02784, TesseraQ 2410.19103, ApiQ 2402.05147, Bias Compensation 2404.01892,
GPTQ-intrinsic LoRA 2606.01412. Mixed precision: SliM-LLM 2405.14917, HIGGS 2411.17525, Q-Palette
2509.20214, APTQ 2402.14866, ternary Qwen3 2609.01962 and 2609.09240, HPTQ 2507.18553, llama.cpp's
`llama_tensor_get_type_impl`. Dissociation and protocol: 2407.09141, 2504.04823, PriDe 2309.03882,
2406.03009, 2606.19558, 2609.07664, 2608.06564, 2609.07901, 2609.01587, 2604.19884, 2506.12044.
The 2026-05 to 2026-09 sweep found no paper reporting MMLU ≥ 60 on Qwen3-4B at ≤ 2.5 b/weight
without training; the LLVQ paper's 60.7 and 62.8 remain the only such numbers, and our 56.37 sits
above every non-LLVQ PTQ figure seen for Qwen3-4B or 8B near 2 bits.
