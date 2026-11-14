# CERVEAU: the state of scientific understanding

Progress: all sections drafted on 2026-11-12. The same day the per-type restore of the sealed 4B updated Q10, Q11,
Q13 and Q16, added Q22, and closed O1; C7 records the code ceiling. Its journal is now in `mesures/`, and Q4, Q20 and
Q21 were reread. An independent review the same day added the standing `proved` and regraded nine claims against
the definitions. 59 claims: 13 established, 2 proved, 35 supported, 6 conjecture, 3 refuted. Claims whose Checked line
says "copied from `ETAT.md`" still need a reread against their journal; A (Metal) is the thinnest section.
On 2026-11-13 the R0 of the quality phase (O12) reread the Q claims against their journals: Q2, Q5, Q6, Q7, Q9, Q10,
N1 and N2 gained scope notes. The same day R1 replayed N1 on the sealed 4B: Q7 was restated and is established on
that object, and a card job added Q23. A first re-encode under a final-output objective then added Q24 and E3. 62
claims: 14 established, 2 proved, 37 supported, 6 conjecture, 3 refuted.

This document lists what we believe to be true about quantizing LLMs on the Leech lattice, each belief with its
standing. It is not the served state ([`ETAT.md`](ETAT.md)), the history ([`HISTORIQUE.md`](HISTORIQUE.md)) or a plan
([`ROADMAP.md`](ROADMAP.md)). Both working modes of [`MODES.md`](MODES.md) read it first and write it last.
The journals in [`mesures/`](mesures/) stay the record; this file says which of their
results still hold.

## How to read a claim

Each claim has a stable number (Q7, K3) that journals cite. It carries five fields.

- **Standing.** `established`: measured on the served object, and replicated or controlled. `proved`: the claim
  follows from a proof or an exhaustive enumeration, and a test of the repository or a cited derivation locks it; it
  carries no measurement. `supported`: measured once, or on an older object (Planes14, a pre-seal base), a smaller
  model or synthetic data. `conjecture`: computed or estimated, never measured. `refuted`: a measurement contradicts
  it, named in the claim. `superseded`: another claim replaced it, named in the claim.
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
- Checked: 2026-11-13. The journal's own reserve, lines 138 to 142: the seeds drew random offsets and the published
  file a contiguous prefix, so the 2.92 does not attach to the published file's 55.59.

**Q3.** At constant file, the bar is the paired interval, not the draw sd. On the full split one MMLU pair has an SE
of 0.17 to 0.38 points. On 2,280 q one MMLU pair had an SE of 0.43 points, 95 % interval [−0.45, +1.22] around
+0.33. One perplexity pair had a 95 % interval of half-width 0.12 %, [−0.0713, +0.1703] % around +0.0494 %.
- Standing: supported. Scope: 4B. The 2,280 q and perplexity figures are one pair (KV f16 against q8) on the Planes14
  file, Metal, f16. The full-split SEs are pairs on the sealed 4B and its parent.
- Evidence: [kvq8-4b](mesures/kvq8-4b-2026-08-15.txt) line 73 (Δ +0.33 pp, IC95 [−0.45, +1.22], SE 0.43) and lines
  52 to 53 (Δ ppl +0.0494 %, IC95 [−0.0713, +0.1703]); [sealed-rownorms](mesures/sealed-rownorms-2026-09-23.txt)
  (SE 0.17); [embed-q4-swap](mesures/embed-q4-swap-2026-09-23.txt) (SE 0.31 and 0.38).
- Overturn: none expected; this is a property of the instrument.
- Checked: 2026-11-12.

**Q4.** Under about 0.4 dB (*estimated* threshold, drawn from three arms), a better Gaussian MSE of the codebook has
given a worse perplexity (*measured*), and its MMLU effect is not resolved. At the same ideal rate of 2.0702,
`leech0c13`, Tetra and Planes14 read 19.61, 16.16 and 16.94 of perplexity and 54.67, 53.49 and 55.59 of MMLU.
- Standing: supported. Scope: 4B, pre-row-scale Tetra-era bases, C4 calibration, 2,280 q, one encoding per arm.
- Evidence: [leech0c13](mesures/leech0c13-2026-09-07.txt). Tetra against Planes14 on MMLU: −2.11 [−5.04, +0.81],
  unresolved ([q5-apparie](mesures/q5-apparie-2026-09-09.txt)).
- Overturn: a codebook change under 0.4 dB that moves paired MMLU outside its interval in the predicted direction
  on two draws.
- Checked: 2026-11-12. The dB figures and the threshold are in the study note
  [innovation-tetra-v2-forme](mesures/innovation-tetra-v2-forme-2026-11-11.txt), sections 1 and 3. On Gaussian blocks,
  against Tetra, `leech0c13` has the same
  MSE (*reported*, paper Table 8), Planes14 a lower MSE by 0.38 dB (*measured*) and the Euclidean gain rule by 0.08 dB.
  None of the three carried its sign to perplexity. The 0.4 dB line is that note's reading of three arms
  (*estimated*), not a fit, and stays the claim's weakest part.

**Q5.** Perplexity is a weak proxy for MMLU here, and the two can move in opposite resolved directions. A gain-map
correction cut perplexity by 17.4 % and cost 3.06 MMLU points, p = 6.2e-18.
- Standing: supported (measured on the C4 base before row scales, not the served object). Scope: 4B Tetra, C4 base
  before row scales, full split. Four other cases point the same way:
  Q4, Q7, Q17, and the spherical feedback of Q18.
- Evidence: [errmap-mmlu-4b](mesures/errmap-mmlu-4b-2026-09-16.txt) (16.80 → 13.88 ppl, 51.25 → 48.19 MMLU).
  Against AWQ on sealed files, perplexity reads our 4B 6.9 % below AWQ while MMLU reads it 4.76 below (`ETAT.md` §3).
- Overturn: a set of arms where paired perplexity orders MMLU correctly over several resolved pairs.
- Checked: 2026-11-13. On the sealed 4B, NLL moves against KL and against MMLU on the same arm twice: restoring q, k
  and v to FP16 raises the sealed model's NLL by 0.040 [0.037, 0.043] while its KL to FP16 falls by 0.025 and MMLU
  rises by +0.55; sealing q, k and v alone on an FP16 body lowers NLL by 0.072 while its KL to FP16 is 0.075
  ([layer-vs-output-4b](mesures/layer-vs-output-4b-2026-11-13.txt), *measured*).
  The witness is `sealed-temoin.bin`, 51.25 on the full split, where bare Tetra on C4 reads 54.64.
  The journal states neither the witness's calibration nor its relation to that file
  ([recherche-qualite-r0-depot](mesures/recherche-qualite-r0-depot-2026-11-13.txt) section 9).

**Q6.** KL to the dense model is also a weak handle: one global temperature removes 2.49 % of the held-out KL on
WikiText-2 and 0.93 % on C4, and buys no MMLU.
- Standing: supported. Scope: 4B Tetra without corrections, Metal f32, 4,092 held-out positions.
- Evidence: [kl-temperature-4b](mesures/kl-temperature-4b-2026-09-16.txt) (T* = 1.07216).
- Overturn: a KL-trained correction whose MMLU gain is predicted by its KL gain.
- Checked: 2026-11-13. "Buys no MMLU" holds by construction (*computed*): a positive temperature cannot move an
  argmax, and the journal runs no MMLU arm (its lines 43 to 52). The claim stands for a global temperature and for
  training on KL (Q12). It is not a statement on ranking: at group grain KL ordered the restore groups as MMLU does
  (Q23).

**Q7.** Quantization costs MMLU beyond the amplitude of its logit error, and on the sealed 4B the amplitude carries
most of the gap. iid Gaussian noise of the sealed 4B's own residual amplitude on the four f16 option logits predicts
64.65 against 63.37 measured on the full split: −1.27 [−1.89, −0.65]. FP16 reads 70.14, so the amplitude alone predicts
5.49 of the 6.77 points and 1.27 lie beyond it. On bare Tetra, 2,280 q, the same noise predicts 58.53 against 53.49.
- Standing: established for the sealed 4B's gap, which ran under controls (rule D of its prereg); supported for the
  older arms. Scope: 4B. The sealed 4B on 14,042 q, f16 dense reconstruction, one calibration draw. Older arms
  (Planes14, bare Tetra, three seeds) on 2,280 q. The gap is to a noise model on four logits and is conditional on the
  fitted slope and residual sd.
- Evidence: [recherche-qualite-n1](mesures/recherche-qualite-n1-2026-11-13.txt) (prereg sha256 3323f313; sealed 4B
  G_gauss −1.27, *computed* on *measured* dumps); [logit-snr-4b](mesures/logit-snr-4b-2026-09-12.txt) section 5 (gaussian iid
  58.53 ± 1.05 against 53.49).
- Overturn: a second draw of the sealed 4B, or another instrument than MMLU, where the Gaussian interval contains 0.
  The former condition, the replay on the sealed 4B, did not fire on 2026-11-13.
- Former statement, until 2026-11-13: "The error is structured: where it lands matters more than its size." It does
  not hold as a statement of shares on the sealed 4B, where size predicts 5.49 points and structure 1.27.
- Checked: 2026-11-13. The f16 margin does not carry the sealed 4B's 1.27: reshuffled inside 20 bands of margin the
  arm's own residuals leave −1.43 [−2.03, −0.82]. The gap is not LLVQ-only: on the full split AWQ 4B reads −0.88
  [−1.31, −0.44]. Descriptive, one draw each: bare Tetra −4.45 [−5.16, −3.74], the sealed 8B −2.33 [−2.87, −1.78], the
  sealed 14B −1.38 [−1.84, −0.92]. The instrument does not see a constant offset on one option letter. On the older
  arms: the noise is iid on the four option logits of the f16 dump, added offline by
  `ops/logit_snr.py` at the arm's fitted slope and residual sd; no weight and no activation is perturbed. The ± 1.05
  is the sd over simulation draws, and the journal gives no interval on the 5-point gap. Tetra's own residuals
  reshuffled across questions predict 58.76, and reshuffled inside 20 bands of f16 margin 57.53. The six LLVQ arms
  all read under their Gaussian prediction; on that 2,280-q sample AWQ reads over it, 70.04 against 69.21 ± 0.83
  ([recherche-qualite-r0-depot](mesures/recherche-qualite-r0-depot-2026-11-13.txt) section 3).

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
- Checked: 2026-11-13. "Flip" here is a reversed shape, a correlation below zero between the arm's four centred
  logits and f16's, not a changed pick. The share depends on that cut: 5.5 % of questions for Tetra under −0.5 and
  29.8 % under +0.5 ([selftest](mesures/logit-snr-selftest-2026-09-12.txt) lines 36 to 42). The five encodings were
  made separately but share the encoder drift of 2026-08-26, and Planes14 and Tetra differ in codebook.

**Q10.** On the Planes14 file, gains from restoring projection types are sub-additive. The seven single f16 restores
sum to 23.85 points, all seven together give 14.73; the sets they repair overlap at 22 to 36 % Jaccard against 3 to
6 % by chance. On the sealed 4B this is not reproduced at group level. All 252 projections give +6.72, more than the MLP plus
attention (5.583) by +1.14 [+0.43, +1.85]. Attention is worth +0.73 [+0.16, +1.29] on the sealed MLP and +1.87
[+1.35, +2.36] on an f16 MLP.
- Standing: supported, on the Planes14 file. Scope: published Planes14 4B file, 2,280 q, constant-file A/B, seven
  single types. The sealed reading covers four groups (qkv, o, gate/up, down), the full split and one draw.
- Evidence: [m2-attribution-4b](mesures/m2-attribution-4b-2026-09-02.txt); Jaccard in
  [logit-snr-4b](mesures/logit-snr-4b-2026-09-12.txt) section 7. The sum of the seven gains, 23.85, against 14.73 for
  all seven is in [m2-attribution-4b](mesures/m2-attribution-4b-2026-09-02.txt) line 136 (*computed* there).
  Sealed 4B: [ablation-qualite-4b](mesures/ablation-qualite-4b-2026-11-12.txt), section ADDITIVITY. The
  excess of all over mlp plus attn is *computed* per question; the gains are *measured*. S, all over the four single
  groups, reads 1.157 [0.973, 1.457] (*computed*, bootstrap), not resolved from 1, against M2's 0.618 on seven types.
  Inside each group the deviation is not resolved: mlp −0.53 [−1.21, +0.15], attn +0.31 [−0.12, +0.74]. M2 summed
  seven types and S four groups, so part of M2's interaction sits inside the groups here. Across the same two groups
  M2 was itself sub-additive: all minus attention plus MLP reads −2.95 on the Planes14 file and −1.55 on its seed-3
  copy ([m2rep](mesures/m2rep-graine3-4b-2026-09-04.txt)), both *computed* in
  [recherche-qualite-r0-depot](mesures/recherche-qualite-r0-depot-2026-11-13.txt) section 4c, with no interval. The
  cross-group sign differs between the two objects. In KL on running text the sealed 4B is super-additive across
  attention and MLP on both sides: all minus attention plus MLP reads +0.0524 [+0.0499, +0.0548] nats with a group
  restored, and +0.0529 [+0.0505, +0.0554] with a group sealed on an FP16 body (*computed*,
  [layer-vs-output-4b](mesures/layer-vs-output-4b-2026-11-13.txt), ADDITIVITY).
- Overturn: on its own scope, a rerun where the seven single types sum inside the interval of the joint arm.
- Checked: 2026-11-13.

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
- Standing: supported. Scope: the DCLM parent bases at three sizes, before the int4 swap and the q4 embedding that
  made the sealed files, full split, paired; the second pass is on the sealed 4B.
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
  the q4 embedding at about 0.35 ([innovation-ecart-qualite](mesures/innovation-ecart-qualite-2026-11-12.txt), a
  study note, section 1).

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
- Evidence: [innovation-ecart-qualite](mesures/innovation-ecart-qualite-2026-11-12.txt) (verdict);
  [gains-criteres-qualite](mesures/gains-criteres-qualite-2026-11-11.txt) (best candidate 1 to 2 of 6.77). Both are
  study notes, and their figures *estimated*.
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

**Q22.** Per billion restored weights, the MLP's rate equals the rate of all projections, and all projections are
mostly MLP weights. The whole MLP recovers +1.81 MMLU points per billion weights, all 252 projections +1.85, gate/up
+1.89. Attention reads lower, +0.77 as a group, qkv +0.97 and o −0.34; down reads higher, +2.24. The MLP carries
71.7 % of the gap with 2.690 of the 3.633 billion restored weights. The rates carry no interval, so whether any group
is a hot spot is not resolved.
- Standing: supported. The gains are measured once on the served object; the rates are *computed* from them and the
  journal carries no interval on a rate. Scope: as Q16. A rate is per weight restored to f16, not per b/param of a
  fix.
- Evidence: [ablation-qualite-4b](mesures/ablation-qualite-4b-2026-11-12.txt), RESULT, columns "gain per Gw" and
  "share of 6.77" (*computed*), restored weights from control 5; reading in point 6.
- Overturn: per-weight rates with intervals that separate one group's rate from the rate of all projections.
- Checked: 2026-11-12.

**Q23.** On the sealed 4B, at group grain, KL to FP16 on running text orders the four projection groups as the MMLU
restores do, and the relative layer-output error does not: it rates gate/up lowest. Restoring a group recovers, of
KL(T || S) = 0.2852 nats a token, gate/up 36.29 %, down 23.17 %, qkv 8.74 %, o 2.48 %, the order of the MMLU gains
(+3.38, +2.01, +0.55, −0.13), with down above qkv by +0.0411 [+0.0372, +0.0451]. The layer error pooled per group
reads down > qkv > o > gate/up with FP16 or sealed inputs, gate/up at 0.0072 and 0.0071.
- Standing: supported. Measured once, on the served object, under a stamped prereg whose primary readings it settles
  (outcome R-a). Four groups are four points. Scope: sealed 4B, its f16 export, one calibration draw, WikiText-2
  test, 73 windows of 4,096, torch f16 on an L40S (perplexity within 0.033 % of the candle path). Group grain only;
  the 252 per-matrix values are written and not read. The layer error compared across groups is not what the encoder
  does, which minimises inside each matrix: the claim says nothing on what an output-aware objective would gain.
- Evidence: [layer-vs-output-4b](mesures/layer-vs-output-4b-2026-11-13.txt), RESULT and READING (*measured*, job
  6af6f9da, $0.44); [ablation-qualite-4b](mesures/ablation-qualite-4b-2026-11-12.txt) for the MMLU gains.
- Overturn: another draw, size or text where the KL order of the four groups departs from the MMLU order.
- Checked: 2026-11-13.

**Q24.** Encoding gate_proj and up_proj of the 4B under gradient-weighted input Hessians (GuidedQuant's objective,
G = 4, weights from the encoder's own 64 windows) does not beat the layer objective on the same draw. Unclipped it
loses: 57.11 against 58.62 on the full MMLU split, paired −1.51 [−2.30, −0.70]. With each group's weights clipped at
10 times their mean it reads 57.95, −0.68 [−1.47, +0.14] against the baseline and +0.83 [+0.03, +1.65] against the
unclipped encode. WikiText-2 perplexity: 16.46 baseline, 18.08 unclipped, 16.69 clipped.
- Standing: supported. Two pairs under two stamped preregs (LOSS, then NOT RESOLVED), three card encodes, same card
  model, same tokens, one variable each. Scope: 4B, one calibration draw, bases without row scales or q4 embedding,
  G = 4, two matrix types, one clip level chosen from the weight files after the first result. One draw cannot
  attribute a difference to the objective alone. The unclipped weights are concentrated (*computed*): effective
  sample size 1.58 % of the 131,072 tokens, about 2,070, under the 2,560 inputs of the matrix; the clip raises it to
  17.3 %. That the concentration caused the first loss is supported by the secondary pair, whose interval starts at
  +0.03. The two-sided form (YAQA) did not run.
- Evidence: [guided-gateup-4b](mesures/guided-gateup-4b-2026-11-13.txt) and
  [guided-clip-gateup-4b](mesures/guided-clip-gateup-4b-2026-11-14.txt) (*measured*, six card jobs, $13.05).
- Overturn: a second draw, another clip level or more calibration windows where the guided encode beats its baseline
  with a resolved pair.
- Checked: 2026-11-14.

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
- Evidence: [large-4e-hors-poids-etude](mesures/large-4e-hors-poids-etude-2026-11-12.txt), a study note, section 1
  (*computed*); [large-4e-e1](mesures/large-4e-e1-2026-11-11.txt) lines 141 to 144 (*measured*).
- Overturn: a per-phase DRAM counter trace showing idle bandwidth in the prefetching phases.
- Checked: 2026-11-12.

**K6.** The 4B door has an estimated ceiling of 235 to 258 tok/s at the bit and 241 to 286 off the bit (within the
±0.12 % perplexity band); ×1.3 over v9 is out of reach either way.
- Standing: conjecture. Scope: 4B, L40S, batch 1.
- Evidence: [`plan-300.md`](plan-300.md) (at the bit);
  [large-4e-hors-poids-etude](mesures/large-4e-hors-poids-etude-2026-11-12.txt) section 3 (off the bit).
- Overturn: a measured 4E form above 258 at the bit or 286 off it.
- Checked: 2026-11-12.

**K7.** The batched GEMM is far from bandwidth-bound: at B = 32 the 4B moves 106.4 GB/s against 428.3 for AWQ in vLLM,
and the projections are three quarters of the step.
- Standing: supported. Scope: 4B, L40S, kernel alone; GB/s *computed* from measured tok/s.
- Evidence: [batch-l40s-v2](mesures/batch-l40s-v2-2026-11-08.txt), [batch-l40s](mesures/batch-l40s-2026-11-08.txt).
- Overturn: a batched Tetra GEMM above 300 GB/s at B = 32.
- Checked: 2026-11-12 (copied from `ETAT.md` §4).

**K8.** CUDA Graphs buy +12.6 % of throughput for +47 % of VRAM at 4B; the trade is refused in the core.
- Standing: supported, as a closed lead. Scope: the Planes14 4B in the served config of 2026-08-31, before Tetra and
  the sealed files, L40S.
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
- Standing: supported (synthetic data, not the served object). Scope: synthetic Gaussian, 20,000 evaluation blocks,
  two seeds.
- Evidence: [gainrule-tetra](mesures/gainrule-tetra-2026-09-12.txt) (arm A, arm D); paper Table 3 in
  [`llvq-paper-notes.md`](llvq-paper-notes.md).
- Overturn: none; the instrument is deterministic per seed.
- Checked: 2026-11-12.

**C2.** The blocks are close to their lattice direction: cos θ has mean 0.9608, sd 0.0077; the norm rule pays 1.0200
over the continuous optimum.
- Standing: supported (synthetic data, not the served object). Scope: same as C1, synthetic Gaussian.
- Evidence: [gainrule-tetra](mesures/gainrule-tetra-2026-09-12.txt).
- Overturn: none.
- Checked: 2026-11-12.

**C3.** Tetra, Planes14 and `leech0c13` encode the 4B at the same ideal rate, 2.0702 bits, and differ only in VRAM
layout and codebook; Tetra reads 2.150 kernel b/weight against 4.804 for Planes14.
- Standing: proved for the ideal rate and for Planes14's 4.804. Tetra's 2.150 is locked by no test: it was computed
  before the bench and read there at the upload (*measured*). Scope: 4B; kernel b/weight from
  `docs/data/echelle-formats.csv`.
- Evidence: the ideal rate is 48 bits a block plus the f16 tail, `Report::bits_per_weight`, 2.070226
  ([`fiche-4b.md`](fiche-4b.md) §5.3); the three codebooks read it identically to the fourth decimal
  (`HISTORIQUE.md` 2026-09-07). The test `published_4b_kernel_rates_are_reproduced` in
  `llvq-bench/src/bin/rtbits.rs` rebuilds 4.804 from block counts alone. Tetra: predicted 2.150 [2.149, 2.151], read
  2.150 ([banc-252](mesures/banc-252-2026-10-06.txt), PREDICTIONS).
- Overturn: none for the proved part; a test that rebuilds Tetra's kernel rate and disagrees with 2.150.
- Checked: 2026-11-12.

**C4.** Whole-model b/param of the sealed files, int4 tables included, is 2.73 / 2.70 / 2.73 at 4B / 8B / 14B.
- Standing: proved (an exhaustive count of each file's bytes by `rtbits`, recorded in the sealing journals). Scope:
  the three sealed files.
- Evidence: `ETAT.md` §2. `rtbits` over each file's bytes reads 2.7320 at 4B
  ([embed-q4-swap ECARTS](../proofs/preregistration-embed-q4-swap-2026-09-23-ECARTS.md) §É1), 2.6953 at 8B
  ([sealed-8b-27](mesures/sealed-8b-27-2026-09-24.txt), file B) and 2.7305 at 14B
  ([sealed-8b-14b](mesures/sealed-8b-14b-2026-09-23.txt)). The test
  `whole_model_conversion_matches_the_published_8b_cells` in `llvq-bench/src/bin/rtbits.rs` locks the whole-model
  conversion, not these three values. In llama.cpp's file-byte accounting the same files read 2.82 / 2.75 / 2.76
  ([duel-iq2](mesures/duel-iq2-2026-11-08.txt)).
- Overturn: a recount that disagrees with `rtbits`.
- Checked: 2026-11-12.

**C5.** Denser VRAM layouts of the same codes stop paying before b_max: Golay70 reached 1.77× against a 2.0× threshold,
and E3 needs 3.0444 kernel b/weight against a 2.60 criterion.
- Standing: supported, as closed leads. Scope: layouts that are not served; L40S bench (Golay70), computed (E3).
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
- Evidence: [innovation-tetra-v2-forme](mesures/innovation-tetra-v2-forme-2026-11-11.txt), a study note, verdict and
  sections 1 and 2.
- Overturn: a 24-dimension fixed-rate code of 48 bits measured more than 0.60 dB under Tetra.
- Checked: 2026-11-12.

## S. How it scales with model size

**S1.** The gap to FP16 shrinks with size: 6.77, 5.48 and 3.22 MMLU points, and 9.63, 4.62 and 3.26 GSM8K points, at
4B, 8B and 14B.
- Standing: established. Scope: the three sealed files, full splits, paired. The order holds on the point values; it
  is not resolved everywhere. The 8B and 14B GSM8K intervals overlap, [3.00, 6.25] and [1.93, 4.59], and the 4B and
  8B MMLU intervals touch, [6.05, 7.50] and [4.83, 6.10].
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
- Evidence: [large-4e-hors-poids-etude](mesures/large-4e-hors-poids-etude-2026-11-12.txt), a study note, section 1.
- Overturn: a KV-read change that moves the phase by much less than its byte change.
- Checked: 2026-11-12.

**P4.** An int8 KV cache does not pass the served tolerance at long context: its own measured interval leaves the
±0.12 % band.
- Standing: supported. Scope: 4B; the KV journal is Metal, short context, on the Planes14 file.
- Evidence: [kvq8-4b](mesures/kvq8-4b-2026-08-15.txt); the band reading from
  [large-4e-hors-poids-etude](mesures/large-4e-hors-poids-etude-2026-11-12.txt), section 2, cut C3.
- Overturn: an int8 or int4 KV arm inside ±0.12 % at 8,192 on the sealed file.
- Checked: 2026-11-12 (band reading not checked against the kvq8 figures).

## D. Speculative decoding

**D1.** With a 0.6B drafter and 4E verifying one row at a time, speculation is slower than greedy at every size:
×0.595, ×0.629, ×0.625 at the best k.
- Standing: established, as a closed lead. Scope: sealed 4B/8B/14B, L40S, tokens identical to greedy.
- Evidence: [spec-gate](mesures/spec-gate-2026-11-07.txt).
- Overturn: the reopening condition of `ETAT.md` §6.
- Checked: 2026-11-12.

**D2.** A tree verify pass prices an extra row at 0.0114 to 0.0196 of a 4E token, but its fixed part is large. The
repaired pass of 2026-11-10 costs 1.544 4E tokens for one row and 1.728 for 16 rows. The earlier S1 pass of 2026-11-09
cost 2.798 for one row.
- Standing: supported. Scope: sealed 4B, L40S, chat context.
- Evidence: [verify-tree-n11](mesures/verify-tree-n11-2026-11-10.txt), arm a10t384r16 (c_1 1.544 [1.539-1.547], c_16
  1.728 [1.724-1.734], extra row 0.0123); [verify-tree-s1](mesures/verify-tree-s1-2026-11-09.txt) (c_1 2.798, extra
  row 0.0196).
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
- Evidence: [served-v9](mesures/served-v9-2026-11-07.txt) and its 8B and 14B siblings (card after load, 1.980 /
  3.356 / 5.637 GB, *measured*). The weights are *computed* by the door from the tensors it loads, as `ETAT.md` §2
  states; the served-v9 journals do not carry them.
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
- Evidence: [innovation-cout-encodage](mesures/innovation-cout-encodage-2026-11-12.txt), a study note, section 1;
  row-scale training at 4B in [dclm-rowscales](mesures/dclm-rowscales-2026-09-20.txt) (1.71 h in the loop, about
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

**E3.** A card encode of the 4B at today's recipe takes 77 to 81 billed minutes on `rtx-pro-6000` and costs $3.55 to
$3.71; quantization, which is CPU work on 23 threads, is 58 minutes of it. The Mac encode of the same recipe is 1 h 47.
- Standing: supported (*measured* three times, 2026-11-13 and 2026-11-14; costs *computed* at $2.75/h). Scope: 4B, 64
  windows of 2,048, `smoke` then `seal`, GPU peak 29.2 GiB of 96, host peak 23.1 GB. No 8B or 14B card encode of
  this recipe is timed.
- Evidence: [guided-gateup-4b](mesures/guided-gateup-4b-2026-11-13.txt), section on the card encode;
  [guided-clip-gateup-4b](mesures/guided-clip-gateup-4b-2026-11-14.txt).
- Overturn: a card encode of the same recipe outside 70 to 90 minutes.
- Checked: 2026-11-13.

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
| N1 | The quantization error costs more MMLU than Gaussian noise of the same amplitude on the four option logits. On the sealed 4B, full split: −1.27 [−1.89, −0.65], 1.27 of the 6.77 points to FP16. AWQ 4B reads −0.88 on the same split. | Q7, and Q9 for the shared fragile questions. | Which property of the error does the damage. The f16 margin does not carry it on the sealed 4B. |
| N2 | Attention is worth more once the MLP is restored than on the sealed MLP: the groups are super-additive. Sealed 4B, one draw, MMLU only; both Planes14 draws were sub-additive across the same groups. | Q10. | Why the errors of the two groups compound on one object and overlap on the other. |
| N3 | A lower Gaussian MSE of the codebook has given a worse perplexity twice, Planes14 at 4B and the Euclidean gain rule at 0.6B; no MMLU pair between codebooks is resolved. | Q4, Q17. | What the model is sensitive to that block MSE does not see. |
| N4 | Perplexity and KL close while MMLU stays open. | Q5, Q6. | What MMLU reads that the two do not. |
| N5 | The Tetra row loop streams faster alone than inside 4E, and bit-identical cuts move time without removing it. | K4, K10; K5 for the split of the time without weights, which is a conjecture (its consequence, the 94 % that reappeared, is measured). | What holds the token when the rows are not the limit. |

## Open questions

Ranked by how much they block.

**O1.** Closed 2026-11-12. Where the sealed 4B's 6.77 MMLU points go, group by group. Job 6af550a4 answered it:
the MLP carries +4.86 of the gap, attention +0.73, and all projections restored sit +0.04 [−0.23, +0.31] under FP16
(Q16, Q22; [ablation-qualite-4b](mesures/ablation-qualite-4b-2026-11-12.txt)). What remains open is O9.

**O2.** Whether an output-aware encoder objective gains on MMLU and GSM8K (Q20). No arm has ever run; it is the only
quality lever with a projected gain above 1 point.

**O3.** How much of the 2.92-point draw spread survives row scales (Q2). A second draw of a sealed size, through row
scales, on the full split, settles it; until then every cross-file quality comparison carries it.

**O4.** What an off-the-bit 4E buys at the door (K5, K6). The desk study
[large-4e-hors-poids-etude](mesures/large-4e-hors-poids-etude-2026-11-12.txt) is done; its first experiment, 4E
v9as with a teacher-forced perplexity mode, about $0.60, is not run.

**O5.** The served scores through the served kernel. MMLU and perplexity ran on the dense reconstruction; GSM8K under
4E has a 50-problem pilot per size only. A census through the kernel settles it.

**O6.** Speed on a consumer card (K3). No sm_86 to sm_120 consumer card is measured.

**O7.** Whether the GSM8K gap (9.63 at 4B) and the MMLU gap share causes. GSM8K was never scored before row scales.

**O8.** The quality of a model above 14B (S3). The 32B must be encoded first.

**O9.** Why attention and the MLP interact on the sealed 4B (Q10). Attention is worth +0.73 on the sealed MLP and
+1.87 [+1.35, +2.36] on an f16 MLP; no mechanism is stated. Unknown, and not chased: how the gains of qkv and down
split between their lattice and int4 records. The operator ruled on 2026-11-12 that per-model, per-type tuning is not
the project's direction.

**O10.** The route each axis allows, as [`MODES.md`](MODES.md) states it on 2026-11-12. Memory, the code on Λ₂₄, has
a ready verifier, and the ceiling of 24-dimension fixed-rate codes is recorded (C7). Speed, the kernel, has a costly
verifier. Model quality has no fast, faithful verifier: it is open through the explanation route, from the anomalies
N1 and N2.

**O12.** Phase open: MODE RECHERCHE on model quality, opened by the operator on 2026-11-13. It runs beside O11, the
phase on decode speed, which is on the branch `recherche/debit-2026-11-12`. Its mandate:

| field | content |
|---|---|
| Axis | Quality of the served object at its b/param: MMLU on the full split and GSM8K, paired against FP16. Whole-model b/param, VRAM, decode speed, time to first token and the cost of producing an object are constraints. |
| Starting point | The anomaly N1, read where Q16 puts the gap: the MLP carries +4.86 of the 6.77 points, gate/up +3.38. N1 is measured on older arms only (Q7: bare Tetra, 2,280 q), so R1 first settles whether it holds on the sealed 4B. N2 is admissible if R1 finds it the sharper problem. |
| Untouchable | The three sealed files. Whole-model b/param at 2.73. VRAM at the served peaks. No edit under `llvq-cuda/` or to a served config while O11 is open: that phase works there. A per-model or per-type setting is not a result (operator, 2026-11-12), and a restore to f16 is a probe, not a mechanism (Q20). |
| Budget | Five days, to 2026-11-18. $10 of card under a cap of its own, apart from the $25 cap of 2026-11-09 that O11 draws on; each job under a stamped prereg and the operator's go. No model loaded on the work machine while O11 is open. |

The route is the explanation route (O10). Two research phases run together under the rule of
[`MODES.md`](MODES.md) as the operator amended it the same day.

State on 2026-11-13: R0 is done, and R1's first measurement is done, $0 of $10 spent. N1 holds on the sealed 4B at
−1.27 [−1.89, −0.65] MMLU points, and the amplitude of the logit error alone predicts 5.49 of the 6.77
([recherche-qualite-n1](mesures/recherche-qualite-n1-2026-11-13.txt), Q7). The gap N1 offers the phase is therefore
0.65 to 1.89 points at 4B. The operator ruled the same day: the phase drops N1 as its problem and takes the amplitude
share, the 5.49 points. His reason: where an error lands against chance cannot be known in advance, so it is not a
place to work. This amends the mandate's starting point, which is now the gap between the Gaussian prediction, 64.65,
and FP16, 70.14, on the sealed 4B. Axis and untouchables are unchanged. The $10 budget is short for this problem,
where an attempt needs an encode and a full MMLU arm; the lead brings a costed plan before any spend. N1 stays in
the anomaly list, set aside. The framing of the new starting point is
[recherche-qualite-r1-cadrage](mesures/recherche-qualite-r1-cadrage-2026-11-13.txt): no proxy in the repository
has ranked encodings as MMLU does, so a candidate is scored by a full MMLU arm, $0.66 to $0.72; one 4B attempt
costs $10.37 to $11.83 on the card alone and $4.69 with the encode on the Mac, $0.69 without row scales (*computed*
there). The Mac clause of the mandate is the operator's to keep or lift. On the operator's go under $5, a first
measurement ran on the card for $0.44
([layer-vs-output-4b](mesures/layer-vs-output-4b-2026-11-13.txt)): at group grain the final-output judge follows the MMLU truth and the layer judge does not (Q23). Spent: $0.44 of $10. No encoder has
been changed and no gain is measured. The operator then gave the go for a first re-encode under a final-output
objective, in its smallest form: gradient-weighted input Hessians for gate_proj and up_proj, scored by one full MMLU
pair against a baseline encoded in the same run
([recherche-qualite-r2-chiffrage-encodeur](mesures/recherche-qualite-r2-chiffrage-encodeur-2026-11-13.txt), V1,
$1.65 to $1.89 *estimated*). He lifted the mandate's Mac clause for the two encodes of this attempt: one at a time,
at low priority, never beside another heavy job. The statistic job ran for $0.09. The operator then moved the two
encodes to the card and raised the phase's budget from $10 to $15: a card encode of today's recipe has never been
timed, and its price is wanted for the larger sizes. The image is built from this branch in a separate private
Space, so the image the speed phase pins does not move. The attempt then ran in full
([guided-gateup-4b](mesures/guided-gateup-4b-2026-11-13.txt)): LOSS under its prereg, candidate minus baseline −1.51
[−2.30, −0.70] on the full split, perplexity +9.86 % (Q24). A card encode of the 4B is 77 to 80 minutes and $3.55 to
$3.68 (E3). Read after the result, the candidate's weights rest on about 2 % of the calibration tokens, a candidate
cause. On the operator's go of 2026-11-14 one retry ran with the weights clipped at 10 times the group mean
([guided-clip-gateup-4b](mesures/guided-clip-gateup-4b-2026-11-14.txt)): NOT RESOLVED against the baseline, −0.68
[−1.47, +0.14], and +0.83 [+0.03, +1.65] against the unclipped encode. By its prereg this closes the gradient-weighted
input-Hessian form for the phase. Spent: $13.49 of $15. The $1.51 left pays no encode. What comes next is the
operator's call: nothing is launched.
R0's repository half is
[recherche-qualite-r0-depot](mesures/recherche-qualite-r0-depot-2026-11-13.txt), the literature half
[recherche-qualite-r0-litterature](mesures/recherche-qualite-r0-litterature-2026-11-13.txt). Three facts frame R1.
N1's Gaussian arm is on the four option logits, not on weights (Q7). Its replay on the sealed 4B needs dumps only,
and they are in the repository. No published study answers N1 or N2; the nearest finds that quantization shrinks
decision margins by a factor with a shared offset (*reported*, literature note section 1a). The mandate's +4.86 is
the MLP restored on sealed attention; on f16 attention the MLP is worth +6.00 [+5.31, +6.69]
([ablation-qualite-4b](mesures/ablation-qualite-4b-2026-11-12.txt) line 126).

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
| Group restores are sub-additive on the sealed 4B | 2026-11-12 | across groups only: all exceeds mlp + attn by +1.14 [+0.43, +1.85]. S predicted 0.77 [0.60, 0.95], measured 1.157 [0.973, 1.457], not resolved from 1; inside the MLP the ratio 0.90 [0.795, 1.031] is not resolved (same journal; Q10) |

## Disagreements found between documents

- `docs/ETAT.md` §4, paragraph Noise, says the draw sd comes from "three full runs". The journal
  [bruit-mmlu-graines-4b](mesures/bruit-mmlu-graines-4b-2026-08-25.txt) line 38 scores the three seeds on the same
  2,280 questions, not the full split, and its 5.2 % perplexity σ comes from
  [f5-graines-4b](mesures/f5-graines-4b-2026-08-19.txt) (its line 148), not from the same three runs.
- `docs/ETAT.md` §6 cites the Euclidean gain rule verdict to `HISTORIQUE.md` only; the journal
  [tetrapost-ppl-0.6b](mesures/tetrapost-ppl-0.6b-2026-09-15.txt) exists and is the stronger source.
- `docs/ETAT.md` §2 reads 8.6 tok/s for the CPU path; [cpu-tetra](mesures/cpu-tetra-2026-11-07.txt) reads 8.65.
  This is rounding, recorded for completeness.
- `docs/ETAT.md` §4, paragraph Noise, calls 0.43 pp of MMLU "the bar" of an A/B on one file. The journal
  [kvq8-4b](mesures/kvq8-4b-2026-08-15.txt) line 73 gives 0.43 pp as the SE of one pair, whose 95 % interval is
  [−0.45, +1.22]. The 0.12 % of perplexity beside it is a half-width (Q3).
- [embed-q4-swap](mesures/embed-q4-swap-2026-09-23.txt) line 30 gives the sealed 4B 2.7138 b/param, the prereg's
  figure. The prereg's [ECARTS](../proofs/preregistration-embed-q4-swap-2026-09-23-ECARTS.md) §É1 corrects it to
  2.7320, and `ETAT.md` §2's 2.73 follows the correction (C4).
- `docs/METHODE.md` line 67 calls 0.43 pp of MMLU "the paired interval" of an A/B on one file. It is the SE of one
  pair, as for `ETAT.md` above ([kvq8-4b](mesures/kvq8-4b-2026-08-15.txt) line 73).
- `docs/data/jobs.csv`, row `ablation-qualite-4b`, says the prereg's interaction row did not fire. The journal's
  CLARIFICATION, lines 205 to 219, says it fires to the letter and was set aside by judgement.
