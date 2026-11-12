# State of scientific understanding

Progress: written Q (quality), K (decode kernel), C (code); to write: S (size), P (prefill), D (speculative), M
(memory), E (cost), X (other methods), A (Metal), open questions, beliefs we dropped, disagreements.

This document lists what we believe to be true about quantizing LLMs on the Leech lattice, each belief with its
standing. It is not the served state ([`ETAT.md`](ETAT.md)), the history ([`HISTORIQUE.md`](HISTORIQUE.md)) or a plan
([`ROADMAP.md`](ROADMAP.md)). The journals in [`mesures/`](mesures/) stay the record; this file says which of their
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
- Evidence: [bruit-mmlu-graines-4b](mesures/bruit-mmlu-graines-4b-2026-08-25.txt) (s = 2.92 pp); the three seeds read
  58.02, 52.19 and 55.17 in [logit-snr-4b](mesures/logit-snr-4b-2026-09-12.txt).
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
- Checked: 2026-11-12. The 0.4 dB threshold is the brief's, not a journal's figure; see Disagreements.

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

**Q10.** Gains from restoring projection types do not add. The seven single f16 restores sum to 23.85 points, all seven
together give 14.73; the sets they repair overlap at 22 to 36 % Jaccard against 3 to 6 % by chance.
- Standing: supported. Scope: published Planes14 4B file, 2,280 q, constant-file A/B. Not measured on the sealed file.
- Evidence: [m2-attribution-4b](mesures/m2-attribution-4b-2026-09-02.txt); Jaccard in
  [logit-snr-4b](mesures/logit-snr-4b-2026-09-12.txt) section 7. The 23.85 and 14.73 are *computed* from the first in
  the branch note `innovation-ecart-qualite-2026-11-12.txt` (`claude/arbre-prefill-2026-11-11`, uncommitted).
- Overturn: the per-type restore of the sealed 4B (open question O1).
- Checked: 2026-11-12.

**Q11.** Spending bits on int4 records pays at 4B. `v_proj` at int4 g128 gives +3.47 [+1.42, +5.57] for +0.0493
b/param; `o_proj` +1.55 [+0.91, +2.20] and `down_proj` +4.05 [+3.28, +4.83] on held-out questions.
- Standing: supported. Scope: 4B Tetra on C4, before row scales; v_proj on 2,280 q, o and down on the 11,762
  questions outside the selection. The served 4B carries v, o and down at layers 12 to 23 at int4.
- Evidence: [q5-tetra](mesures/q5-tetra-2026-09-06.txt), [oproj-int4-full](mesures/oproj-int4-full-2026-09-17.txt),
  [downproj-int4-full](mesures/downproj-int4-full-2026-09-18.txt).
- Overturn: the sealed 4B's per-type restore showing the int4 types carry a share far from these gains (O1).
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

**Q16.** About 83 % of the 4B gap to FP16 (5.6 of 6.77 MMLU points) is attributed by no measurement on the sealed
object; the int4 records account for an estimated 0.5 to 0.8 and the q4 embedding about 0.35, unresolved.
- Standing: conjecture. Scope: sealed 4B; the shares are transferred by arithmetic from older bases.
- Evidence: `innovation-ecart-qualite-2026-11-12.txt`, branch `claude/arbre-prefill-2026-11-11`, uncommitted, section 1.
- Overturn: O1 below.
- Checked: 2026-11-12.

**Q17.** Rescaling the gain of each block toward its projection lowers Gaussian MSE (retention 88.73 → 89.43 %) and
does not help end to end. The Euclidean gain rule is dead.
- Standing: refuted, as a quality lever. Scope: synthetic Gaussian for the MSE, 4B end to end.
- Evidence: [gainrule-tetra](mesures/gainrule-tetra-2026-09-12.txt); end-to-end verdict in `HISTORIQUE.md`
  2026-09-15 and `ETAT.md` §6.
- Overturn: per `ETAT.md` §6, a new idea.
- Checked: 2026-11-12 (MSE side read; end-to-end side from the summary only).

**Q18.** Spherical GPTQ feedback, forming the error against the block at its exact norm, costs +31.89 % of
perplexity.
- Standing: refuted, as a quality lever. Scope: Qwen3-0.6B, one seed, Mac.
- Evidence: [sph-ppl-0.6b](mesures/sph-ppl-0.6b-2026-09-22.txt) (41.89 → 55.25).
- Overturn: per `ETAT.md` §6, a new idea.
- Checked: 2026-11-12.

**Q19.** Lattice-step distillation (K1: moving codes by one lattice step under a loss) cannot deliver its claimed gain:
one step moves a block by 0.332 of its norm, more than the block's own error of 0.290.
- Standing: refuted, by computation, not by measurement. Scope: Tetra blocks, numpy on the Mac.
- Evidence: `refutation-k1-maths-2026-11-11.txt`, branch `arbre-nuit` worktree, not on `main`.
- Overturn: a measured code-moving fine-tune that gains a resolved point at 4B.
- Checked: 2026-11-12.

**Q20.** No single remaining mechanism is expected to close half the 4B gap (+3.4 MMLU). The best candidate, an
output-aware encoder objective of the YAQA kind, projects +0.5 to +1.7 MMLU.
- Standing: conjecture. Scope: 4B, same format.
- Evidence: `innovation-ecart-qualite-2026-11-12.txt` (verdict), `gains-criteres-qualite-2026-11-11.txt` (best
  candidate 1 to 2 of 6.77), both off `main`.
- Overturn: any output-aware arm, which has never run.
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

## Open questions

Ranked by how much they block. To be completed.

**O1.** Where the sealed 4B's 6.77 MMLU points go, type by type (Q10, Q11, Q16). In flight: job 6af550a4, prereg
`proofs/preregistration-ablation-qualite-4b-2026-11-12.md` on branch `claude/ablation-qualite-2026-11-12`. The control
arm reads 63.37, 14,042 of 14,042 picks equal to the published dump. The other arms are not read.

**O2.** What an off-the-bit 4E buys at the door (K5, K6). In flight as a desk study,
`large-4e-hors-poids-etude-2026-11-12.txt`; its first experiment (4E v9as, about $0.60) is not run.

## Beliefs we dropped

To be completed.

## Disagreements found between documents

To be completed.
