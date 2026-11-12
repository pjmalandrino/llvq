# CERVEAU: the state of scientific understanding

Progress: all sections drafted on 2026-11-12. The same day the per-type restore of the sealed 4B updated Q10, Q11,
Q13 and Q16, added Q22, and closed O1; C7 records the code ceiling. Its journal is now in `mesures/`, and Q4, Q20 and
Q21 were reread. 59 claims: 22 established, 28 supported, 6 conjecture, 3 refuted. Claims whose Checked line says
"copied from `ETAT.md`" still need a reread against their journal; A (Metal) is the thinnest section.

This document lists what we believe to be true about quantizing LLMs on the Leech lattice, each belief with its
standing. It is not the served state ([`ETAT.md`](ETAT.md)), the history ([`HISTORIQUE.md`](HISTORIQUE.md)) or a plan
([`ROADMAP.md`](ROADMAP.md)). Both working modes of [`MODES.md`](MODES.md) read it first and write it last.
The journals in [`mesures/`](mesures/) stay the record; this file says which of their
results still hold.

## How to read a claim

Each claim has a stable number (Q7, K3) that journals cite. It carries five fields.

- **Standing.** `established`: measured on the served object, and replicated or controlled. `supported`: measured
  once, or on an older object (Planes14, a pre-seal base) or a smaller model. `conjecture`: computed or estimated,
  never measured. `refuted`: a measurement contradicts it, named in the claim. `superseded`: another claim replaced
  it, named in the claim.
- **Scope.** The object, size, card and dtype the evidence covers. A claim says nothing outside its scope.
- **Evidence.** The journal and the figure, copied from it.
- **Overturn.** The measurement that would change the standing.
- **Checked.** The date the claim was last read against its evidence.

"Served object" means the sealed files of [`ETAT.md`](ETAT.md) §2. Most quality attribution in this project ran on
older objects; each claim says which. MMLU "2,280 q" is the flat 40-per-subject sample; "full split" is all 14,042.
Notes cited from a branch that is not on `main` are marked with the branch name.

## The update rule

1. A journal that closes a measurement names the claims here that it confirms, weakens or overturns, and the same
   commit edits those claims.
2. A claim is never deleted. It changes standing, and a refuted or superseded claim names what replaced it.
3. A number here is a copy of a journal's number, never a new one. Arithmetic belongs in a journal first.

## Q. What sets quality at 2 to 3 bits

**Q1.** The sealed 4B sits 6.77 MMLU points [6.05, 7.50] and 9.63 GSM8K points [7.69, 11.57] below FP16, paired.
- Standing: established. Scope: sealed 4B, full split, 5-shot, f16 dense reconstruction; GSM8K through the kernel,
  1,319 problems.
- Evidence: [embed-q4-swap](mesures/embed-q4-swap-2026-09-23.txt) (f16 − arm +6.77, SE 0.38);
  [f16-full](mesures/f16-full-2026-09-18.txt) (FP16 70.14); GSM8K pairs in `ETAT.md` §3 from
  [gsm8k-wave1](mesures/gsm8k-wave1-2026-09-26.txt).
- Overturn: a re-encoded 4B at the same b/param that closes either interval.
- Checked: 2026-11-12.

**Q2.** The calibration draw behaves as a hyperparameter: one standard deviation moves MMLU by 2.92 points and
perplexity by 5.2 %.
- Standing: supported. Scope: 4B, three encodings with different seeds, before Tetra and before row scales, 2,280 q.
  Every served size rests on one draw.
- Evidence: [bruit-mmlu-graines-4b](mesures/bruit-mmlu-graines-4b-2026-08-25.txt) (s = 2.92 pp; 58.02, 52.19, 55.17);
  perplexity σ 5.2 % from [f5-graines-4b](mesures/f5-graines-4b-2026-08-19.txt), as that journal's line 148 cites it.
- Overturn: a second draw of a sealed file, after row scales, read on the full split. Row scales may shrink the spread;
  this is not measured.
- Checked: 2026-11-12.

**Q3.** At constant file, the bar is the paired interval, not the draw sd. On the full split one MMLU pair has an SE
of 0.17 to 0.38 points; on 2,280 q the paired bar is 0.43 points of MMLU and 0.12 % of perplexity.
- Standing: established. Scope: 4B, any two arms of one file or two files of one draw.
- Evidence: [kvq8-4b](mesures/kvq8-4b-2026-08-15.txt) (0.43 pp, 0.12 %); [sealed-rownorms](mesures/sealed-rownorms-2026-09-23.txt)
  (SE 0.17); [embed-q4-swap](mesures/embed-q4-swap-2026-09-23.txt) (SE 0.31).
- Overturn: none expected; this is a property of the instrument.
- Checked: 2026-11-12.

**Q4.** A better Gaussian MSE of the codebook does not predict the sign of perplexity or MMLU below about 0.4 dB. At
the same ideal rate of 2.0702, `leech0c13`, Tetra and Planes14 read 19.61, 16.16 and 16.94 of perplexity and 54.67,
53.49 and 55.59 of MMLU.
- Standing: supported. Scope: 4B, pre-row-scale Tetra-era bases, C4 calibration, 2,280 q, one encoding per arm.
- Evidence: [leech0c13](mesures/leech0c13-2026-09-07.txt). Tetra against Planes14 on MMLU: −2.11 [−5.04, +0.81],
  unresolved ([q5-apparie](mesures/q5-apparie-2026-09-09.txt)).
- Overturn: a codebook change under 0.4 dB that moves paired MMLU outside its interval in the predicted direction
  on two draws.
- Checked: 2026-11-12. The dB figures and the threshold are in the note `innovation-tetra-v2-forme-2026-11-11.txt`,
  branch `claude/arbre-nuit-2026-11-10`, sections 1 and 3. On Gaussian blocks, against Tetra, `leech0c13` has the same
  MSE (*reported*, paper Table 8), Planes14 a lower MSE by 0.38 dB (*measured*) and the Euclidean gain rule by 0.08 dB.
  None of the three carried its sign to perplexity. The 0.4 dB line is that note's reading of three arms
  (*estimated*), not a fit, and stays the claim's weakest part.

**Q5.** Perplexity is a weak proxy for MMLU here, and the two can move in opposite resolved directions. A gain-map
correction cut perplexity by 17.4 % and cost 3.06 MMLU points, p = 6.2e-18.
- Standing: established. Scope: 4B Tetra, C4 base before row scales, full split. Four other cases point the same way:
  Q4, Q7, Q17, and the spherical feedback of Q18.
- Evidence: [errmap-mmlu-4b](mesures/errmap-mmlu-4b-2026-09-16.txt) (16.80 → 13.88 ppl, 51.25 → 48.19 MMLU).
  Against AWQ on sealed files, perplexity reads our 4B 6.9 % below AWQ while MMLU reads it 4.76 below (`ETAT.md` §3).
- Overturn: a set of arms where paired perplexity orders MMLU correctly over several resolved pairs.
- Checked: 2026-11-12.

**Q6.** KL to the dense model is also a weak handle: one global temperature removes 2.49 % of the held-out KL on
WikiText-2 and 0.93 % on C4, and buys no MMLU.
- Standing: supported. Scope: 4B Tetra without corrections, Metal f32, 4,092 held-out positions.
- Evidence: [kl-temperature-4b](mesures/kl-temperature-4b-2026-09-16.txt) (T* = 1.07216).
- Overturn: a KL-trained correction whose MMLU gain is predicted by its KL gain.
- Checked: 2026-11-12.

**Q7.** The error is structured: where it lands matters more than its size. Gaussian noise of the measured logit
amplitude predicts 58.53 for bare Tetra, which reads 53.49.
- Standing: supported. Scope: 4B, 2,280 q, older arms (Planes14, bare Tetra, three seeds).
- Evidence: [logit-snr-4b](mesures/logit-snr-4b-2026-09-12.txt) section 5 (gaussian iid 58.53 ± 1.05 against 53.49).
- Overturn: the same replay on the sealed 4B where the Gaussian prediction falls inside the measured interval.
- Checked: 2026-11-12.

**Q8.** The SNR law, acc = 54.60 + 14.60 ln SNR over 28 LLVQ-family arms, ranks arms but misses single arms by up to
3 points; it is a fit, not a predictor of a new method.
- Standing: supported. Scope: 4B, 2,280 q, pre-row-scale arms. AWQ is printed against it, not fitted.
- Evidence: [logit-snr-4b](mesures/logit-snr-4b-2026-09-12.txt) section 2 (seed2 −2.95, Tetra −1.53 under the law).
- Overturn: a held-out arm, fitted before it ran, landing inside ±1 point of the law on two occasions.
- Checked: 2026-11-12.

**Q9.** The loss is a flip on a shared fragile bank of questions. Bare Tetra reverses 15.6 % of four-logit shapes,
and independent encodings flip the same questions at 2.4 to 3.2 times chance.
- Standing: supported. Scope: 4B, 2,280 q, older arms.
- Evidence: [logit-snr-4b](mesures/logit-snr-4b-2026-09-12.txt) section 6 (lifts ×2.38 to ×3.24).
- Overturn: the same count on the sealed 4B with lifts near ×1.
- Checked: 2026-11-12.

**Q10.** On the Planes14 file, gains from restoring projection types are sub-additive. The seven single f16 restores
sum to 23.85 points, all seven together give 14.73; the sets they repair overlap at 22 to 36 % Jaccard against 3 to
6 % by chance. On the sealed 4B this is not reproduced at group level. All 252 projections give +6.72, more than the MLP plus
attention (5.583) by +1.14 [+0.43, +1.85]. Attention is worth +0.73 [+0.16, +1.29] on the sealed MLP and +1.87
[+1.35, +2.36] on an f16 MLP.
- Standing: supported, on the Planes14 file. Scope: published Planes14 4B file, 2,280 q, constant-file A/B, seven
  single types. The sealed reading covers four groups (qkv, o, gate/up, down), the full split and one draw.
- Evidence: [m2-attribution-4b](mesures/m2-attribution-4b-2026-09-02.txt); Jaccard in
  [logit-snr-4b](mesures/logit-snr-4b-2026-09-12.txt) section 7. The 23.85 and 14.73 are *computed* from the first in
  the branch note `innovation-ecart-qualite-2026-11-12.txt` (`claude/arbre-prefill-2026-11-11`, uncommitted).
  Sealed 4B: [ablation-qualite-4b](mesures/ablation-qualite-4b-2026-11-12.txt), section ADDITIVITY. The
  excess of all over mlp plus attn is *computed* per question; the gains are *measured*. S, all over the four single
  groups, reads 1.157 [0.973, 1.457] (*computed*, bootstrap), not resolved from 1, against M2's 0.618 on seven types.
  Inside each group the deviation is not resolved: mlp −0.53 [−1.21, +0.15], attn +0.31 [−0.12, +0.74]. M2 summed
  seven types and S four groups, so part of M2's interaction sits inside the groups here.
- Overturn: on its own scope, a rerun where the seven single types sum inside the interval of the joint arm.
- Checked: 2026-11-12.

**Q11.** Spending bits on int4 records pays at 4B. `v_proj` at int4 g128 gives +3.47 [+1.42, +5.57] for +0.0493
b/param; `o_proj` +1.55 [+0.91, +2.20] and `down_proj` +4.05 [+3.28, +4.83] on held-out questions.
- Standing: supported. Scope: 4B Tetra on C4, before row scales; v_proj on 2,280 q, o and down on the 11,762
  questions outside the selection. The served 4B carries v, o and down at layers 12 to 23 at int4.
- Evidence: [q5-tetra](mesures/q5-tetra-2026-09-06.txt), [oproj-int4-full](mesures/oproj-int4-full-2026-09-17.txt),
  [downproj-int4-full](mesures/downproj-int4-full-2026-09-18.txt).
  On the sealed 4B, restoring the 36 int4 g128 records of `o_proj` to f16 gains −0.13 [−0.46, +0.19] (*measured*,
  full split, dense f16): the int4 `o` records cost nothing resolved. The int4 records of `v_proj` and of `down_proj`
  are not isolated, because the qkv and down arms mix lattice and int4 records
  ([ablation-qualite-4b](mesures/ablation-qualite-4b-2026-11-12.txt), RESULT and point 5).
- Overturn: a measurement on the sealed file where an int4 type costs more against f16 than its lattice records do.
  None is planned (operator, 2026-11-12).
- Checked: 2026-11-12.

**Q12.** Trained per-row scales with frozen codes gain 3.15, 3.29 and 1.67 MMLU points at 4B, 8B and 14B, for zero
bits. A second training pass of row scales and norms gains +0.02 [−0.31, +0.36]: the frozen-code lever is spent.
- Standing: established. Scope: DCLM bases at three sizes, full split, paired.
- Evidence: [dclm-rowscales](mesures/dclm-rowscales-2026-09-20.txt) (McNemar p = 1.8e-25),
  [dclm-8b-rowscales](mesures/dclm-8b-rowscales-2026-09-21.txt), [dclm-14b-rowscales](mesures/dclm-14b-rowscales-2026-09-22.txt),
  [sealed-rownorms](mesures/sealed-rownorms-2026-09-23.txt).
- Overturn: a different objective on the same free parameters that gains a resolved point on the sealed file.
- Checked: 2026-11-12 (4B and the second pass read; 8B and 14B copied from `ETAT.md` §4).

**Q13.** The q4 embedding (tied head) paid for two more int4 tables: +2.26 [+1.66, +2.87] over the 61.11 object.
- Standing: established. Scope: 4B, the sealed object against its parent, full split.
- Evidence: [embed-q4-swap](mesures/embed-q4-swap-2026-09-23.txt) (63.37 against 61.11).
  Against FP16 the int4 g64 embedding and tied head cost +0.04 [−0.23, +0.31], not resolved: FP16 minus the arm with
  all 252 projections restored to f16 (*measured*, [ablation-qualite-4b](mesures/ablation-qualite-4b-2026-11-12.txt)).
- Overturn: none; the sealed object is this arm.
- Checked: 2026-11-12.

**Q14.** More calibration data hurts: four times the DCLM-edu volume costs −1.20 [−2.06, −0.39] MMLU, and perplexity
agrees (16.61 against 16.24).
- Standing: supported. Scope: 4B Tetra DCLM base, before row scales, full split, one draw a side, so the draw sd of Q2
  sits inside the comparison.
- Evidence: [dclm-v4](mesures/dclm-v4-2026-09-18.txt).
- Overturn: a volume increase that gains on two draws.
- Checked: 2026-11-12.

**Q15.** The calibration corpus matters at the scale of the draw: the DCLM-edu base reads 57.95 against 56.37 for C4,
same recipe otherwise.
- Standing: supported. Scope: 4B, Q5 bases, full split, one draw each; the 1.58 gap is smaller than one draw sd (Q2).
- Evidence: [dclm-rowscales](mesures/dclm-rowscales-2026-09-20.txt) (57.95); 56.37 in
  [downproj-int4-full](mesures/downproj-int4-full-2026-09-18.txt).
- Overturn: two draws per corpus.
- Checked: 2026-11-12.

**Q16.** On the sealed 4B the MLP carries most of the gap to FP16. Restoring gate/up to f16 recovers +3.38 MMLU
points [+2.75, +4.02], down +2.01 [+1.46, +2.54], the whole MLP +4.86 [+4.16, +5.57]. Attention recovers +0.73
[+0.16, +1.29] as a group: qkv +0.55 [+0.01, +1.11], o −0.13 [−0.46, +0.19]. All 252 projections recover +6.72
[+6.00, +7.45] and sit +0.04 [−0.23, +0.31] under FP16, so restoring the projections closes the gap within its
interval.
- Standing: established. Scope: sealed 4B, its one calibration draw, MMLU full split, 5-shot, f16 dense
  reconstruction, gains paired against the control by `mmlupair --no-fpc`. GSM8K is not measured. Not isolated:
  `v_proj`, and `down_proj` at layers 12 to 23, because the qkv arm (72 lattice, 36 int4 records) and the down arm
  (24 lattice, 12 int4) mix both record types. qkv's interval clears zero by 0.01 and its McNemar test does not
  (p 0.0508).
- Evidence: [ablation-qualite-4b](mesures/ablation-qualite-4b-2026-11-12.txt), job 6af550a4, RESULT (*measured*).
  The control reads 63.3742 with 14,042 of 14,042 picks equal to the published dump; six of six controls green.
- Overturn: a per-group restore of another draw of the 4B, sealed the same way, whose gains fall outside these
  intervals.
- Checked: 2026-11-12.
- Former statement, conjecture until 2026-11-12, superseded by the measurement above: about 83 % of the gap (5.6 of
  6.77 points) was attributed by no measurement on the sealed object; the int4 records were estimated at 0.5 to 0.8 and
  the q4 embedding at about 0.35 (`innovation-ecart-qualite-2026-11-12.txt`, branch `claude/arbre-prefill-2026-11-11`,
  uncommitted, section 1).

**Q17.** Choosing each block's gain level by projection instead of by norm lowers Gaussian MSE (retention 88.73 →
89.43 %) and costs 4.026 % of perplexity end to end. The rule moves every disagreed block down a level, a 2.9 % radial
shrink.
- Standing: refuted, as a quality lever. Scope: synthetic Gaussian for the MSE; Qwen3-0.6B end to end, one encoding
  a side, Mac.
- Evidence: [gainrule-tetra](mesures/gainrule-tetra-2026-09-12.txt) (arm C);
  [tetrapost-ppl-0.6b](mesures/tetrapost-ppl-0.6b-2026-09-15.txt) (41.8875 against 43.5739), as `HISTORIQUE.md`
  2026-09-15 reports it.
- Overturn: per `ETAT.md` §6, a new idea.
- Checked: 2026-11-12 (end-to-end figure from `HISTORIQUE.md`, journal not reopened).

**Q18.** Spherical GPTQ feedback, forming the error against the block at its exact norm, costs +31.89 % of
perplexity.
- Standing: refuted, as a quality lever. Scope: Qwen3-0.6B, one seed, Mac.
- Evidence: [sph-ppl-0.6b](mesures/sph-ppl-0.6b-2026-09-22.txt) (41.89 → 55.25).
- Overturn: per `ETAT.md` §6, a new idea.
- Checked: 2026-11-12.

**Q19.** Lattice-step distillation (K1: moving codes by one lattice step under a loss) cannot deliver its claimed gain:
one step moves a block by 0.332 of its norm, more than the block's own error of 0.290.
- Standing: refuted, by computation, not by measurement. Scope: Tetra blocks, numpy on the Mac.
- Evidence: [refutation-k1-maths](mesures/refutation-k1-maths-2026-11-11.txt).
- Overturn: a measured code-moving fine-tune that gains a resolved point at 4B.
- Checked: 2026-11-12.

**Q20.** No single remaining mechanism is expected to close half the 4B gap (+3.4 MMLU). The best candidate, an
output-aware encoder objective of the YAQA kind, projects +0.5 to +1.7 MMLU. A restore is not a mechanism: gate/up
restored to f16 recovers half the gap (Q16) by spending 16 bits a weight, so it says where the gap sits, not how to
close it at the same b/param.
- Standing: conjecture. Scope: 4B, same format.
- Evidence: `innovation-ecart-qualite-2026-11-12.txt` (verdict), branch `claude/arbre-prefill-2026-11-11`,
  uncommitted; [gains-criteres-qualite](mesures/gains-criteres-qualite-2026-11-11.txt) (best candidate 1 to 2 of
  6.77). Both are *estimated*.
- Overturn: any output-aware arm, which has never run.
- Checked: 2026-11-12.

**Q21.** The ranking of projection types by restore gain depends on the draw: on a second draw nothing between the two
is resolved and the head swaps. What holds across draws is coarse: MLP above attention, `v_proj` third, `q_proj` and
`k_proj` last.
- Standing: supported. Scope: 4B, Planes14 published file and a seed-3 encoding, 2,280 q, constant-file restores.
- Evidence: [m2rep-graine3-4b](mesures/m2rep-graine3-4b-2026-09-04.txt) (largest difference `gate`, z = −1.68), as
  `HISTORIQUE.md` 2026-09-04 reports it.
- Overturn: a per-type restore on two draws of one sealed size whose order agrees above the noise. The closed O1
  confirmed the coarse part on the sealed 4B, at group level: the MLP above attention (Q16). It restored qkv as one
  group and ran one draw, so it cannot test `v_proj` third, `q_proj` and `k_proj` last, or the draw dependence.
- Checked: 2026-11-12 (from `HISTORIQUE.md`, journal not reopened).

**Q22.** Per billion restored weights the MLP is not special. The whole MLP recovers +1.81 MMLU points per billion
weights, all 252 projections +1.85, gate/up +1.89. The MLP carries 71.7 % of the gap with 2.690 of the 3.633 billion
restored weights, roughly in proportion to its weights; the ablation shows no hot spot. Per weight, down reads +2.24,
qkv +0.97, attention as a group +0.77 and o −0.34.
- Standing: supported. The gains are measured once on the served object; the rates are *computed* from them and the
  journal carries no interval on a rate. Scope: as Q16. A rate is per weight restored to f16, not per b/param of a
  fix.
- Evidence: [ablation-qualite-4b](mesures/ablation-qualite-4b-2026-11-12.txt), RESULT, columns "gain per Gw" and
  "share of 6.77" (*computed*), restored weights from control 5; reading in point 6.
- Overturn: per-weight rates whose intervals separate the MLP's rate from the rate of all projections.
- Checked: 2026-11-12.

## K. The decode kernel

**K1.** The served 4E v9 decodes the 4B at 221.5 tok/s against 129.0 for the eager engine, 1.717× round by round, with
256 of 256 tokens equal; 8B and 14B read 1.384× and 1.375×.
- Standing: established. Scope: sealed files, L40S, batch 1, greedy, 5-token prompt, median of five rounds.
- Evidence: [served-v9](mesures/served-v9-2026-11-07.txt), [-8b](mesures/served-v9-8b-2026-11-07.txt),
  [-14b](mesures/served-v9-14b-2026-11-07.txt).
- Overturn: a rerun at the same door outside the round ranges.
- Checked: 2026-11-12 (copied from `ETAT.md` §4).

**K2.** On the 4B's 252 projections, one Tetra pass takes 3.734 ms against 3.263 for AWQ and 11.022 for FP16.
- Standing: supported. Scope: L40S, kernel bench, not the model.
- Evidence: [banc-252](mesures/banc-252-2026-10-06.txt).
- Overturn: a rerun on the same card outside its range.
- Checked: 2026-11-12 (copied from `ETAT.md` §4).

**K3.** The speed gain is an L40S result. On an A100, Tetra beats our FP16 kernel by 1.11× but not cuBLAS, and the
files decode slower than vLLM's FP16.
- Standing: supported. Scope: one A100, one L40S; no consumer card is measured.
- Evidence: [a100](mesures/a100-2026-10-06.txt).
- Overturn: a consumer card (sm_86 to sm_120) measured at the door.
- Checked: 2026-11-12 (copied from `ETAT.md` §4).

**K4.** Bit-identical rescheduling of 4E is exhausted near v9: v5, the row-loop arms, the guard-free tile and moved
prefetches read ×0.978 to ×1.027 of their reference.
- Standing: established. Scope: 4B, L40S, four closed arms.
- Evidence: [shaker-4e](mesures/shaker-4e-2026-11-05.txt) (×1.001), [step1-micro](mesures/step1-micro-2026-11-07.txt)
  (×1.002), [large-attn-ft-4e](mesures/large-attn-ft-4e-2026-11-11.txt) (×0.997 to ×1.027),
  [large-4e-e1](mesures/large-4e-e1-2026-11-11.txt) (×1.003, ×0.996, ×0.978).
- Overturn: a bit-identical arm above ×1.05.
- Checked: 2026-11-12 (from `ETAT.md` §6).

**K5.** The 1.76 ms of a 4E token spent outside weight rows is mostly not idle DRAM. Its phases carry the L2
prefetches of 1,159 MB of the token's 1,378 MB of weights; only about 265 µs leave the DRAM idle.
- Standing: conjecture for the split; the consequence is measured: removing attention's prefetch cut that phase by
  201 µs and 94 % of it reappeared in o and gate/up.
- Scope: sealed 4B, 4E v9, L40S.
- Evidence: `large-4e-hors-poids-etude-2026-11-12.txt`, worktree `4e-hors-poids`, not on `main` (*computed*);
  [large-4e-e1](mesures/large-4e-e1-2026-11-11.txt) lines 141 to 144 (*measured*).
- Overturn: a per-phase DRAM counter trace showing idle bandwidth in the prefetching phases.
- Checked: 2026-11-12.

**K6.** The 4B door has an estimated ceiling of 235 to 258 tok/s at the bit and 241 to 286 off the bit (within the
±0.12 % perplexity band); ×1.3 over v9 is out of reach either way.
- Standing: conjecture. Scope: 4B, L40S, batch 1.
- Evidence: [`plan-300.md`](plan-300.md) (at the bit); `large-4e-hors-poids-etude-2026-11-12.txt` (off the bit).
- Overturn: a measured 4E form above 258 at the bit or 286 off it.
- Checked: 2026-11-12.

**K7.** The batched GEMM is far from bandwidth-bound: at B = 32 the 4B moves 106.4 GB/s against 428.3 for AWQ in vLLM,
and the projections are three quarters of the step.
- Standing: supported. Scope: 4B, L40S, kernel alone; GB/s *computed* from measured tok/s.
- Evidence: [batch-l40s-v2](mesures/batch-l40s-v2-2026-11-08.txt), [batch-l40s](mesures/batch-l40s-2026-11-08.txt).
- Overturn: a batched Tetra GEMM above 300 GB/s at B = 32.
- Checked: 2026-11-12 (copied from `ETAT.md` §4).

**K8.** CUDA Graphs buy +12.6 % of throughput for +47 % of VRAM at 4B; the trade is refused in the core.
- Standing: established, as a closed lead. Scope: 4B, L40S.
- Evidence: [ECARTS](../proofs/preregistration-a2-a3-geometrie-2026-08-31-ECARTS.md) §É7.
- Overturn: per `ETAT.md` §6.
- Checked: 2026-11-12 (copied from `ETAT.md` §6).

**K9.** Bit identity binds the schedule, not the arithmetic class: the served attention forms change logits and are
held to paired perplexity within ±0.12 %, which they pass by a factor of ten (+0.0095 %, −0.0048 %, +0.0023 % at three
sizes).
- Standing: established. Scope: grouped and kernel attention, three sizes, WikiText-2.
- Evidence: [gqa-ppl](mesures/gqa-ppl-2026-11-01.txt), [gqa-8b](mesures/gqa-8b-2026-11-04.txt),
  [gqa-14b](mesures/gqa-14b-2026-11-04.txt).
- Overturn: an off-bit form inside the band that moves GSM8K by a resolved amount.
- Checked: 2026-11-12 (copied from `ETAT.md` §4).

**K10.** The Tetra row loop is not the limit of 4E: alone at 4E's geometry it streams 379 GB/s, against about 260 GB/s
for the same rows inside 4E; a third of the rows' time is lost in situ, and no single suspect is worth ×1.30.
- Standing: supported. Scope: sealed 4B, 4E v2 era (before v9), L40S; the in-situ figure *computed*.
- Evidence: [pers-rowloop](mesures/pers-rowloop-2026-11-05.txt) (strongest suspect l1w, ×1.214).
- Overturn: a v9 in-situ trace where the rows reach 379 GB/s.
- Checked: 2026-11-12.

## C. The code and its rate-distortion

**C1.** On a Gaussian source at 2.000 b/dim, Tetra's shipped gain rule retains 88.73 % of Shannon (MSE 0.0854); a free
gain bounds it at 92.51 %. The paper reports 89.4 % for spherical shaping and 92.1 % for shape-gain.
- Standing: established. Scope: synthetic Gaussian, 20,000 evaluation blocks, two seeds.
- Evidence: [gainrule-tetra](mesures/gainrule-tetra-2026-09-12.txt) (arm A, arm D); paper Table 3 in
  [`llvq-paper-notes.md`](llvq-paper-notes.md).
- Overturn: none; the instrument is deterministic per seed.
- Checked: 2026-11-12.

**C2.** The blocks are close to their lattice direction: cos θ has mean 0.9608, sd 0.0077; the norm rule pays 1.0200
over the continuous optimum.
- Standing: established. Scope: same as C1.
- Evidence: [gainrule-tetra](mesures/gainrule-tetra-2026-09-12.txt).
- Overturn: none.
- Checked: 2026-11-12.

**C3.** Tetra, Planes14 and `leech0c13` encode the 4B at the same ideal rate, 2.0702 bits, and differ only in VRAM
layout and codebook; Tetra reads 2.150 kernel b/weight against 4.804 for Planes14.
- Standing: established (*computed*). Scope: 4B; kernel b/weight from `docs/data/echelle-formats.csv`.
- Evidence: [leech0c13](mesures/leech0c13-2026-09-07.txt), `ETAT.md` §4.
- Overturn: none; this is arithmetic on the format.
- Checked: 2026-11-12.

**C4.** Whole-model b/param of the sealed files, int4 tables included, is 2.73 / 2.70 / 2.73 at 4B / 8B / 14B.
- Standing: established (*computed* by `rtbits`). Scope: the three sealed files.
- Evidence: `ETAT.md` §2. In llama.cpp's file-byte accounting the same files read 2.82 / 2.75 / 2.76
  ([duel-iq2](mesures/duel-iq2-2026-11-08.txt)).
- Overturn: a recount that disagrees with `rtbits`.
- Checked: 2026-11-12.

**C5.** Denser VRAM layouts of the same codes stop paying before b_max: Golay70 reached 1.77× against a 2.0× threshold,
and E3 needs 3.0444 kernel b/weight against a 2.60 criterion.
- Standing: established, as closed leads. Scope: L40S bench (Golay70); computed (E3).
- Evidence: [golay70-v2](mesures/golay70-v2-sept-bras-2026-08-11.txt), [radixstudy](mesures/radixstudy-x4-2026-08-12.txt).
- Overturn: per `ETAT.md` §6.
- Checked: 2026-11-12 (from `ETAT.md` §6).


**C6.** Gaussian retention overstates the loss on real weights. Tetra's 88.89 % against the ball-12 control's 92.00
implied +8.5 % of MSE, and perplexity moved −4.64 %, the opposite sign.
- Standing: supported. Scope: 4B, Tetra against the published Planes14 encoding, C4 calibration, one draw a side.
- Evidence: `HISTORIQUE.md` 2026-09-06 (the Tetra encoding entry), journal not reopened here.
- Overturn: a second draw a side that reverses the perplexity sign.
- Checked: 2026-11-12.

**C7.** No 24-dimension fixed-rate code of 48 bits a block can sit more than 0.60 dB under Tetra on a Gaussian
source. The sphere-covering bound for a shape-gain code of 47 + 1 bits reads MSE 0.07410 a dimension against Tetra's
0.08506, and no code reaches that bound. Inside the Tetra family the best member at Tetra's decoder cost reads
−0.18 dB, and the best at any decoder cost −0.30 dB.
- Standing: conjecture (bound *computed*, members *measured* on synthetic blocks only). Scope: Gaussian N(0,1),
  2.000 b/dim, 24 dimensions, fixed rate, ideal spherical caps. A 48-dimension pair code has its own bound at
  −0.90 dB; a variable-length code is outside the bound.
- Evidence: `innovation-tetra-v2-forme-2026-11-11.txt`, branch `claude/arbre-nuit-2026-11-10`, verdict and sections
  1 and 2.
- Overturn: a 24-dimension fixed-rate code of 48 bits measured more than 0.60 dB under Tetra.
- Checked: 2026-11-12.

## S. How it scales with model size

**S1.** The gap to FP16 shrinks with size: 6.77, 5.48 and 3.22 MMLU points, and 9.63, 4.62 and 3.26 GSM8K points, at
4B, 8B and 14B.
- Standing: established. Scope: the three sealed files, full splits, paired.
- Evidence: `docs/data/paper2-gaps.csv` and `gsm8kpair` as cited in `ETAT.md` §3.
- Overturn: a fourth size that breaks the order.
- Checked: 2026-11-12 (copied from `ETAT.md` §3).

**S2.** At equal memory, one size up beats AWQ one size down on MMLU: our 8B against AWQ's 4B +1.44 [+0.71, +2.19] in
2.76 against 2.67 GB, our 14B against AWQ's 8B +1.87 [+1.20, +2.57] in 5.04 against 6.10 GB.
- Standing: supported. Scope: post hoc re-analysis of committed dumps, not preregistered.
- Evidence: [sizeup](mesures/sizeup-2026-09-28.txt).
- Overturn: a preregistered pairing that reverses either sign.
- Checked: 2026-11-12.

**S3.** The shared-memory wall of `rot_apply` that barred models above 14B is lifted on the card up to width 25,600,
bit-exact; the 32B has never been encoded, so nothing about its quality is known.
- Standing: supported. Scope: kernel outputs against host bytes, one L40S job.
- Evidence: [lot1-4b](mesures/lot1-4b-2026-11-04.txt) (R0 45.43 µs at 25,600).
- Overturn: a 32B encoded and served.
- Checked: 2026-11-12.

## P. Prefill and long context

**P1.** The served eager prefill is the long-context bottleneck: the 4B's first token at 8,192 takes 35.05 s, while
decode after that prompt still reads 161.5 tok/s.
- Standing: established. Scope: sealed 4B, 4E v9 door, L40S.
- Evidence: [served-v9](mesures/served-v9-2026-11-07.txt).
- Overturn: a served prefill under 10 s at 8,192.
- Checked: 2026-11-12 (copied from `ETAT.md` §4).

**P2.** Feeding the prompt through the tree verify pass (the chain prefill) cuts the 4B's first token to 0.420 / 2.393 /
6.485 s at 1,024 / 4,096 / 8,191 tokens, against 3.951 / 17.641 / 37.037 eager, at +0.0007 % of paired perplexity.
- Standing: supported. Scope: sealed 4B, L40S, two jobs; a bench, not served; GSM8K pairing owed.
- Evidence: [large-prefill-ft](mesures/large-prefill-ft-2026-11-11.txt), [large-prefill](mesures/large-prefill-2026-11-11.txt).
- Overturn: a GSM8K-200 pairing that separates it from eager.
- Checked: 2026-11-12 (copied from `ETAT.md` §5).

**P3.** At long context the decode attention is bound by its own bytes: at 1,024 and 8,192 tokens the attention phase
equals the DRAM time of the KV read plus the prefetch at 85 %, 1,119 against 1,108 µs and 2,559 against 2,646 µs.
- Standing: conjecture (model *computed* against *measured* phases). Scope: sealed 4B, 4E v9, L40S.
- Evidence: `large-4e-hors-poids-etude-2026-11-12.txt`, worktree `4e-hors-poids`, not on `main`.
- Overturn: a KV-read change that moves the phase by much less than its byte change.
- Checked: 2026-11-12.

**P4.** An int8 KV cache does not pass the served tolerance at long context: its own measured interval leaves the
±0.12 % band.
- Standing: supported. Scope: 4B; the KV journal is Metal, short context, on the Planes14 file.
- Evidence: [kvq8-4b](mesures/kvq8-4b-2026-08-15.txt); the band reading from `large-4e-hors-poids-etude-2026-11-12.txt`.
- Overturn: an int8 or int4 KV arm inside ±0.12 % at 8,192 on the sealed file.
- Checked: 2026-11-12 (band reading not checked against the kvq8 figures).

## D. Speculative decoding

**D1.** With a 0.6B drafter and 4E verifying one row at a time, speculation is slower than greedy at every size:
×0.595, ×0.629, ×0.625 at the best k.
- Standing: established, as a closed lead. Scope: sealed 4B/8B/14B, L40S, tokens identical to greedy.
- Evidence: [spec-gate](mesures/spec-gate-2026-11-07.txt).
- Overturn: the reopening condition of `ETAT.md` §6.
- Checked: 2026-11-12.

**D2.** A tree verify pass prices an extra row at 0.0114 to 0.0196 of a 4E token, but its fixed part is large: a
one-row pass costs 2.80 4E tokens, and 16 rows cost 1.73.
- Standing: supported. Scope: sealed 4B, L40S, chat context.
- Evidence: [verify-tree-s1](mesures/verify-tree-s1-2026-11-09.txt) (2.798), [verify-tree-n11](mesures/verify-tree-n11-2026-11-10.txt)
  (c_16 1.728).
- Overturn: a pass whose one-row cost falls under 1.4 tokens.
- Checked: 2026-11-12.

**D3.** Speculation pays on predictable text and barely on chat: the public DFlash drafter with the tree loop decodes
GSM8K-200 at 502.1 tok/s (×2.295), the house set at 343.1 (×1.556), its chat prompts at 241.7 (×1.086), GSM8K 167
correct against 166.
- Standing: supported. Scope: sealed 4B, L40S, one run each.
- Evidence: [spec-loop-sp8](mesures/spec-loop-sp8-2026-11-11.txt) (chat line 116).
- Overturn: a second run outside the round ranges.
- Checked: 2026-11-12.

**D4.** The speed of speculation is paid in memory: 5.41 GB against 2.01 and 4.87 b/param with the f16 drafter, so the
default stays at 2.73 b/param.
- Standing: supported (5.41 *measured*, 4.87 *computed*). Scope: sealed 4B, L40S.
- Evidence: [spec-mem-m3](mesures/spec-mem-m3-2026-11-11.txt), [spec-memoire](mesures/spec-memoire-2026-11-11.txt).
- Overturn: a quantized drafter that keeps the speed.
- Checked: 2026-11-12 (copied from `ETAT.md` §5).

## M. Memory

**M1.** Short context, the served files occupy 1.98 / 3.36 / 5.64 GB of card after load at 4B / 8B / 14B, of which the
weights are 1.38 / 2.76 / 5.04 GB.
- Standing: established. Scope: L40S, `cuMemGetInfo`, v9 doors.
- Evidence: [served-v9](mesures/served-v9-2026-11-07.txt) and its 8B and 14B siblings.
- Overturn: a remeasure outside rounding.
- Checked: 2026-11-12 (copied from `ETAT.md` §2).

**M2.** After an 8,192-token prompt the peak is set by prefill activations, not weights: 5.95 / 7.39 / 10.54 GB, against
3.52 / 4.71 / 6.90 for llama.cpp IQ2_S. Three live f16 score tensors were 6.44 GB of the old 4B peak.
- Standing: established (peaks *measured*, the 6.44 *computed*). Scope: L40S, three sealed files.
- Evidence: [pic-vram-8k](mesures/pic-vram-8k-2026-11-08.txt).
- Overturn: a prefill whose peak exceeds its computed band by more than 0.5 GB.
- Checked: 2026-11-12.

## E. Cost of producing an object

**E1.** A sealed object costs $4.00 / $5.66 / $25.13 of production at 4B / 8B / 14B, and $6.29 / $8.63 / $30.62 with
the MMLU runs of its chain.
- Standing: supported (*measured* rows of `docs/data/jobs.csv`, sums *computed*). Scope: the three sealed chains.
- Evidence: `innovation-cout-encodage-2026-11-12.txt`, branch `claude/arbre-prefill-2026-11-11`, uncommitted, section
  1; row-scale training at 4B in [dclm-rowscales](mesures/dclm-rowscales-2026-09-20.txt) (1.71 h in the loop, about
  $4.00).
- Overturn: a recount of `jobs.csv` that disagrees.
- Checked: 2026-11-12.

**E2.** At 4B and 8B the encode is free on the Mac and every production dollar is the row-scale training; at 14B the
row scales are 39 % of production. No bit-identical lever divides the chain by 3: the best composition reaches about
/1.3 at 4B and 8B and /1.8 to /2.1 at 14B.
- Standing: conjecture (*computed* from measured phases). Scope: the three sealed chains; a 32B is *estimated* at $57
  to $107.
- Evidence: same note, verdict and section 1.
- Overturn: a measured encode or training run that beats the computed composition.
- Checked: 2026-11-12.

## X. Against other methods

**X1.** At the same bits, Tetra beats llama.cpp IQ2_S on quality and loses on speed: +9.08, +4.13, +2.89 MMLU at 4B,
8B, 14B, while IQ2_S decodes 60.1, 63.9 and 33.5 tok/s faster after 5 tokens.
- Standing: established. Scope: same day, same L40S, imatrix on dclm-edu, llama-bench accounting.
- Evidence: [duel-iq2](mesures/duel-iq2-2026-11-08.txt).
- Overturn: a newer llama.cpp quant at our bits within the MMLU interval.
- Checked: 2026-11-12 (copied from `ETAT.md` §3).

**X2.** Against AWQ w4g128 we lose 4.76 / 4.21 / 2.46 MMLU points at 45 to 52 % of its bits per parameter.
- Standing: established. Scope: three sizes, vLLM AWQ, full split, paired.
- Evidence: `docs/data/paper2-gaps.csv` as cited in `ETAT.md` §3.
- Overturn: a re-encoding that closes the interval.
- Checked: 2026-11-12 (copied from `ETAT.md` §3).

**X3.** The source paper's `leech0c13` 4B reads 60.7 MMLU in the paper and 54.67 here on 2,280 q; the gap is not the
codebook (Q4), and the paper's recipe is not reproduced.
- Standing: supported. Scope: 4B, C4 calibration here.
- Evidence: [leech0c13](mesures/leech0c13-2026-09-07.txt).
- Overturn: our encoder at the paper's recipe reaching 60.7 within its interval.
- Checked: 2026-11-12.

## A. Metal and MLX

**A1.** The MLX door reads −0.0103 % [−0.0847, +0.0642] of paired perplexity against the Rust dense reconstruction,
and its prototype decodes 93.9 tok/s on an M3 Max where mlx-lm's 4-bit 4B decodes 129.5; the Tetra kernel is the gap.
- Standing: supported. Scope: 4B, M3 Max, f16, other stacks for mlx-lm.
- Evidence: [mlx-pooled](mesures/mlx-pooled-2026-11-07.txt), [metal-nuit](mesures/metal-nuit-2026-11-06.txt) M16.
- Overturn: a Tetra MLX kernel at or above 129.5.
- Checked: 2026-11-12 (copied from `ETAT.md` §4).

**A2.** On a CPU, reading the codes at every matvec beats the dense reconstruction by speed and memory: 8.65 tok/s in
1.86 GB against 0.71 tok/s in 10.6 GB, perplexity −0.0021 %.
- Standing: supported. Scope: 4B, M3 Max, 12 threads, f32; never run on x86.
- Evidence: [cpu-tetra](mesures/cpu-tetra-2026-11-07.txt).
- Overturn: an x86 run.
- Checked: 2026-11-12.

## Anomalies

A measured fact that no claim explains. They are the starting points of the explanation route of
[`MODES.md`](MODES.md). An anomaly leaves this list when a claim explains it, and that claim says so.

| | anomaly | where it is measured | what is missing |
|---|---|---|---|
| N1 | The quantization error costs more MMLU than Gaussian noise of the same amplitude. | Q7, and Q9 for the shared fragile questions. | Which property of the error does the damage. |
| N2 | Attention is worth more once the MLP is restored than on the sealed MLP: the groups are super-additive. | Q10, sealed 4B. | Why the errors of the two groups compound. |
| N3 | A lower Gaussian MSE of the codebook does not give a better model, and has given a worse one. | Q4, Q17. | What the model is sensitive to that block MSE does not see. |
| N4 | Perplexity and KL close while MMLU stays open. | Q5, Q6. | What MMLU reads that the two do not. |
| N5 | The Tetra row loop streams faster alone than inside 4E, and bit-identical cuts move time without removing it. | K4, K5, K10. | What holds the token when the rows are not the limit. |

## Open questions

Ranked by how much they block.

**O1.** Closed 2026-11-12. Where the sealed 4B's 6.77 MMLU points go, group by group. Job 6af550a4 answered it:
the MLP carries +4.86 of the gap, attention +0.73, and all projections restored sit +0.04 [−0.23, +0.31] under FP16
(Q16, Q22; [ablation-qualite-4b](mesures/ablation-qualite-4b-2026-11-12.txt)). What remains open is O9.

**O2.** Whether an output-aware encoder objective gains on MMLU and GSM8K (Q20). No arm has ever run; it is the only
quality lever with a projected gain above 1 point.

**O3.** How much of the 2.92-point draw spread survives row scales (Q2). A second draw of a sealed size, through row
scales, on the full split, settles it; until then every cross-file quality comparison carries it.

**O4.** What an off-the-bit 4E buys at the door (K5, K6). Desk study `large-4e-hors-poids-etude-2026-11-12.txt` is
done; its first experiment, 4E v9as with a teacher-forced perplexity mode, about $0.60, is not run.

**O5.** The served scores through the served kernel. MMLU and perplexity ran on the dense reconstruction; GSM8K under
4E has a 50-problem pilot per size only. A census through the kernel settles it.

**O6.** Speed on a consumer card (K3). No sm_86 to sm_120 consumer card is measured.

**O7.** Whether the GSM8K gap (9.63 at 4B) and the MMLU gap share causes. GSM8K was never scored before row scales.

**O8.** The quality of a model above 14B (S3). The 32B must be encoded first.

**O9.** Why attention and the MLP interact on the sealed 4B (Q10). Attention is worth +0.73 on the sealed MLP and
+1.87 [+1.35, +2.36] on an f16 MLP; no mechanism is stated. Unknown, and not chased: how the gains of qkv and down
split between their lattice and int4 records. The operator ruled on 2026-11-12 that per-model, per-type tuning is not
the project's direction.

**O10.** The three research fronts, as [`MODES.md`](MODES.md) states them on 2026-11-12. The code on Λ₂₄ has a ready
verifier, and the ceiling of 24-dimension fixed-rate codes is recorded (C7). The kernel has a costly verifier. Model
quality has no fast, faithful verifier, so it is closed to research until one exists.

## Beliefs we dropped

| belief | dropped | by |
|---|---|---|
| Quantizing the gain costs almost nothing | 2026-07-31 | the A/B compared an arm with itself; +3.17 % ppl for −0.618 b/weight (`HISTORIQUE.md`) |
| The first 4B reads 2.07 to 2.11 b/weight | 2026-07-31 | the 16-bit magnitude was not charged; 2.73 (`HISTORIQUE.md`) |
| In-domain calibration flatters perplexity by 12 % | 2026-08 | C4 gives 14.91 against 15.29 on WikiText (`fiche-4b.md`) |
| A single shell beats the union | 2026-08-04 | union 0.0725 against shell 13 at 0.0762 (`HISTORIQUE.md`) |
| Perplexity noise across draws is 0.7 % | 2026-08-19 | σ 5.2 % ([f5-graines-4b](mesures/f5-graines-4b-2026-08-19.txt)) |
| MMLU noise across draws is 0.5 to 1.5 points | 2026-08-25 | s = 2.92 ([bruit-mmlu](mesures/bruit-mmlu-graines-4b-2026-08-25.txt)) |
| All format work is capped at 4.77× | 2026-08-21 | nullk is the floor of our geometry (`HISTORIQUE.md`) |
| k_proj and attention are the sensitive types | 2026-09 | v_proj, 2.6 % of weights (`HISTORIQUE.md`; [m2-attribution](mesures/m2-attribution-4b-2026-09-02.txt)) |
| The paper's codebook explains our 5.1-point MMLU gap | 2026-09-07 | [leech0c13](mesures/leech0c13-2026-09-07.txt) |
| Tetra is worse than Planes14 on MMLU | 2026-09-09 | −2.11 [−5.04, +0.81], never resolved ([q5-apparie](mesures/q5-apparie-2026-09-09.txt)) |
| The Euclidean gain rule helps end to end | 2026-09-15 | +4.026 % ppl at 0.6B (Q17) |
| KL to the dense model is dominated by scale | 2026-09-16 | one temperature removes 0.93 to 2.49 % ([kl-temperature](mesures/kl-temperature-4b-2026-09-16.txt)) |
| A better perplexity correction helps MMLU | 2026-09-16 | −17.4 % ppl, −3.06 MMLU ([errmap-mmlu](mesures/errmap-mmlu-4b-2026-09-16.txt)) |
| More calibration volume helps | 2026-09-18 | −1.20 [−2.06, −0.39] ([dclm-v4](mesures/dclm-v4-2026-09-18.txt)) |
| Spherical GPTQ feedback helps | 2026-09-22 | +31.89 % ppl ([sph-ppl-0.6b](mesures/sph-ppl-0.6b-2026-09-22.txt)) |
| Tetra loses to FP16 on an A100 | 2026-10-06 | it beats our FP16 kernel by 1.11× ([a100](mesures/a100-2026-10-06.txt)) |
| Speculation is closed on verify cost | 2026-11-10 | an extra tree row costs 0.0114 to 0.0196 of a token (D2) |
| Moving 4E's prefetch frees time | 2026-11-11 | 94 % of the time reappears downstream (K5) |
| Lattice-step distillation closes a large share of the gap | 2026-11-11 | computed refutation (Q19) |
| Attention carries a large share of the sealed 4B's gap | 2026-11-12 | predicted qkv +1.4, o +0.8, attn +2.0; measured +0.55, −0.13, +0.73, all below their intervals ([ablation-qualite-4b](mesures/ablation-qualite-4b-2026-11-12.txt); Q16) |
| Group restores are sub-additive on the sealed 4B | 2026-11-12 | S predicted 0.77 [0.60, 0.95], measured 1.157; all exceeds mlp + attn by +1.14 [+0.43, +1.85] (same journal; Q10) |

## Disagreements found between documents

- `docs/ETAT.md` §4, paragraph Noise, says the draw sd comes from "three full runs". The journal
  [bruit-mmlu-graines-4b](mesures/bruit-mmlu-graines-4b-2026-08-25.txt) line 38 scores the three seeds on the same
  2,280 questions, not the full split, and its 5.2 % perplexity σ comes from
  [f5-graines-4b](mesures/f5-graines-4b-2026-08-19.txt) (its line 148), not from the same three runs.
- `docs/ETAT.md` §6 cites the Euclidean gain rule verdict to `HISTORIQUE.md` only; the journal
  [tetrapost-ppl-0.6b](mesures/tetrapost-ppl-0.6b-2026-09-15.txt) exists and is the stronger source.
- `docs/ETAT.md` §2 reads 8.6 tok/s for the CPU path; [cpu-tetra](mesures/cpu-tetra-2026-11-07.txt) reads 8.65.
  This is rounding, recorded for completeness.
