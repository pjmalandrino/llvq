# Where the quality goes: the logit-fidelity accounting

Every MMLU bar this repository has published throws away 99% of what its job measured. `mmlu`
writes the four option logits verbatim so that a later analysis can rank them
(`llvq-llm/src/bin/mmlu.rs:448`); the accuracy keeps one argmax per question. This document
reads the logits back, on the 37 dumps already on disk. It is *computed* throughout, on
*measured* dumps, and cost $0. The accounting is reproduced by `uv run ops/logit_snr.py`, whose
output is the journal [logit-snr-4b-2026-09-12](mesures/logit-snr-4b-2026-09-12.txt). No run was
started and no decision is taken here.

The estimator is the repository's own: accuracy stratified by subject population, which
reproduces 70.32, 70.04, 55.59, 53.49, 56.95, 58.02, 52.19 and 55.17 to the hundredth, and their
published bars to 0.01 pp.

## The model, in one line

For a question, centre the four f16 logits on their own mean as `x` and the arm's as `y`, then
regress `y = beta * x + e` over the 9,120 values of the bank. Two numbers come out: `beta`, the
share of the reference signal that survives quantization, and `sd(e)`, what replaces it. Their
ratio is the fidelity, `SNR = beta * sd(x) / sd(e)`.

| arm | beta | sd(e) | SNR | noise share of the served logit variance | MMLU |
|---|---|---|---|---|---|
| AWQ w4 g128 | 0.960 | 0.856 | 3.656 | 7.0% | 70.04 |
| `Planes14`, published | 0.466 | 1.418 | 1.071 | 46.6% | 55.59 |
| `Tetra` | 0.458 | 1.451 | 1.029 | 48.6% | 53.49 |
| `Tetra` + `v_proj` int4 | 0.498 | 1.443 | 1.125 | 44.1% | 56.95 |

Read the third column first. Almost half of the logit variance the served 4B produces is
quantization noise, and just under half of the reference signal survives. AWQ, at a rate 1.9
times ours, loses 4% of the signal and carries 7% of noise.

## One scalar accounts for the whole family

Fit `acc = a + c ln(SNR)` on 28 arms: two codebooks, three calibration draws, seven projection
types restored to f16, two restored to int4, two aggregates, two base files, two seeds of the
attribution campaign. The fit is `acc = 54.60 + 14.60 ln(SNR)` with an rms residual of 1.31 pp,
against a per-arm sampling error of 1.41 pp.

The relation is partly mechanical, since both quantities are read on the same four logits. What
is measured is the absence of a second axis. Read the SNR on 28 subjects and the accuracy on the
other 29, and the law holds out of sample at 1.74 pp of rms residual, where the arms spread over
3.33 pp and one arm's sampling error on 29 subjects is 1.59 pp. At equal logit fidelity, none of
our design choices produces a better MMLU. The codebook does not, which is what the
`leech0c13` campaign of 2026-09-07 found the expensive way. Neither does the calibration draw,
the restored matrix, nor the format.

The law prices the work. One MMLU point costs a factor 1.071 on the SNR, that is a division of
the noise-to-signal ratio by 1.15. Reading 65 at the 4B needs the SNR of the served `Tetra` arm
multiplied by 1.98, that is its noise power divided by 3.9.

## The error grows with the reference model's confidence

Split the bank by the f16 top-two margin and take the per-question rms residual in each decile.

| decile | f16 margin | AWQ | `Planes14` | `Tetra` |
|---|---|---|---|---|
| 1 | 0.00 to 0.55 | 0.709 | 0.997 | 0.985 |
| 5 | 3.28 to 4.67 | 0.834 | 1.258 | 1.133 |
| 10 | 10.00 to 13.59 | 0.505 | 1.607 | 1.681 |

Our per-question error rises with the f16 margin, at a correlation of +0.16 to +0.32 across five
independent encodings. AWQ's falls, at −0.18. A shape-gain code carries this by
construction: the reconstruction error of a block is proportional to its norm, so the output
error follows the amplitude of what the model was surest of. A scalar quantizer with a per-group
absmax bounds its error by a step size instead, and AWQ's channel scaling protects the salient
directions on top of that.

Its measured cost is small. Replay each arm's own residual vectors on the f16 logits, reshuffled
inside bands of equal f16 margin, and the accuracy moves by 0.2 to 1.0 pp against a free
reshuffle. The mechanism is worth more than its direct cost, because it says where the bits
should go.

## A reversal on part of the bank

Take the correlation between the four f16 logits of a question and the four the arm produces.
For AWQ, 0.7% of the bank comes out negative. For the five LLVQ encodings, 11.1 to 15.6%. On
those questions the arm scores 10.8 to 15.2%, below the 25% of a coin, while f16 scores 53.5 to
62.4%. The arm answers something else on those questions, and holds it as firmly as f16 held the
right one.

A homogeneous noise of the same distribution produces 6.6 to 8.3% of such questions, so 4 to 7
points of the reversal are coupled to the question itself. That coupling is what separates the
measured accuracy from the amplitude accounting: an iid gaussian of the same variance predicts
59.6 for `Planes14` and 58.5 for `Tetra`, against 55.59 and 53.49 measured. Replaying each arm's
own residuals, reshuffled, lands in the same place. Three to five points of MMLU are therefore
carried by where our error falls rather than by how large it is.

Across encodings the broken sets overlap at 2.4 to 3.2 times chance, so the fragility is partly a
property of the input and partly a draw. Their union over the five encodings covers 35.5% of the
bank. One encoding in five breaks a third of what the model knows, and each one breaks a
different third. That is the mechanism behind the 2.92 pp of MMLU sigma across calibration draws,
and behind three formats that span 21.5% of perplexity while their MMLU stays inside one interval.

The five encodings are independent by calibration draw or by codebook. They are not independent
of the encoder drift of 2026-08-26 (`docs/ETAT.md` §4): the three seeds predate it, `Tetra` does
not.

## Attribution by projection type ranks noise

Restoring one projection type to f16 repairs a set of questions. The sets overlap at a Jaccard of
22.0 to 35.8% where chance gives 3.2 to 5.6%, and the union of the seven single-type sets is 489
questions where f16 itself repairs 490. Every type repairs the same pool. Restoring a type does
not give back a faculty; it lifts the fidelity, and the questions nearest the threshold come back.

The right quantity to rank types is therefore what each contributes to the noise-to-signal ratio,
which is stable where the MMLU ranking is not.

| type | noise share, seed 0 | seed 3 | weight share | noise per weight, seed 0 | seed 3 |
|---|---|---|---|---|---|
| `down_proj` | 44.7% | 46.1% | 24.7% | 1.578 | 1.444 |
| `v_proj` | 29.7% | 24.7% | 2.6% | 9.962 | 7.357 |
| `up_proj` | 29.2% | 33.5% | 24.7% | 1.032 | 1.049 |
| `gate_proj` | 26.1% | 27.9% | 24.7% | 0.923 | 0.873 |
| `o_proj` | 31.1% | 10.6% | 10.4% | 2.608 | 0.793 |
| `k_proj` | 6.7% | 10.2% | 2.6% | 2.261 | 3.049 |
| `q_proj` | 10.7% | 3.5% | 10.4% | 0.898 | 0.261 |

`down_proj` is first on both draws and `v_proj` carries 7 to 10 times the noise per weight of any
MLP matrix. The MMLU ranking moved `gate_proj` from first to fourth between the same two draws
(`docs/ETAT.md` §5 ter); the noise ranking does not move for those two types. The shares sum to
157 to 178% of the total, and the aggregates measure a sub-additivity of 0.68 to 0.90 for
attention and 0.73 to 0.78 for the MLP, which matches the 0.618 to 0.792 already on record.

`down_proj` leading is consistent with two facts already in the file. Its input is
`act(gate) * up`, the one input `calib.rs:705-720` captures before quantizing anything inside the
block (row C of [ROADMAP-QUALITY](ROADMAP-QUALITY.md)). Its input dimension is 9,728, where the
massive activations live (row E).

## Attenuation is not a calibration defect

A `beta` of 0.466 reads like a squashed output that one temperature would undo. The best
temperature on the four MMLU letters is 2.38 for f16 and 2.13 to 2.56 for the quantized arms, and
applying the best temperature to both sides leaves the excess NLL where it was, or widens it by
up to 64%. AWQ is the only arm where a temperature buys anything, 52% of a much smaller excess.

The reason is in the simulation. Attenuation alone, at `beta = 0.466` with no noise, moves the
four-way NLL from 1.0218 to 0.7513, an improvement, and leaves the accuracy at 70.27 because an
argmax ignores a positive scale. Noise alone at the same ratio gives 2.09 of NLL and 59.96 of
accuracy. The served arm reads 1.2128 and 55.59. The attenuation is already the temperature. The
model has become less confident in proportion to the noise it carries, which is what keeps its
perplexity from being far worse than it is.

This contradicts row B of [ROADMAP-QUALITY](ROADMAP-QUALITY.md), which reads the 0.4660 slope as
output calibration recoverable at zero bits. On these logits a folded temperature would cost
accuracy nothing and NLL something. The reservation is the measurement's own: it covers four
letter tokens at one position, not the 151k-token vocabulary of a perplexity window.

## Across sizes

| size | beta | sd(e) | SNR | MMLU loss against f16 |
|---|---|---|---|---|
| 4B | 0.466 | 1.418 | 1.071 | 14.73 pp |
| 8B | 0.748 | 1.612 | 1.590 | 10.57 pp |
| 14B | 0.812 | 1.272 | 2.210 | 6.85 pp |

AWQ holds beta at 0.956 to 0.971 at all three sizes. Ours climbs from 0.466 to 0.812. The gain
with size is a fidelity gain, and it is in the signal term rather than the noise term. The
published loss curve is that fidelity curve read through the law, with a ceiling that differs per
size.

## What this does not establish

The fidelity is measured on four letter tokens at the last position of a 5-shot prompt. Nothing
here measures the full vocabulary, and the relation between this SNR and wikitext perplexity is
not established: the ratio of excess NLL to noise-to-signal runs 0.37, 0.50 and 0.85 from 4B to
14B, so it is not one constant.

No measurement here is per layer. The dumps restore a type across all 36 blocks at once, so
nothing separates depth from type, and nothing says whether the noise accumulates or is injected
late.

The law is fitted on arms whose SNR runs 1.03 to 2.43. AWQ at 3.66 sits 3.48 pp under its
extrapolation, which is expected of a logarithm against a ceiling of 70.32 and is not evidence
about the law inside its range.

The 4 to 7 points of excess reversal are a coupling between the error and the question. This
document does not say what the coupling is. The obvious candidate, that the broken questions
activate directions the calibration prefix never covered, is untested and testable for $0 on the
artifact already on disk.
