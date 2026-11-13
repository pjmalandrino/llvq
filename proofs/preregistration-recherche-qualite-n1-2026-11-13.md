# Preregistration. Anomaly N1 on the sealed 4B: measured MMLU against iid Gaussian noise of its own logit amplitude

**Written on 2026-11-13, BEFORE the first measurement. NOT YET STAMPED.** The signed predictions (§9) are filled by the
main session, then the file is stamped (`ots stamp`) and committed before `ops/logit_n1_replay.py controls` or `run`
reads any full-split dump. Once stamped, this file is not edited; a deviation goes into `-ECARTS.md` beside it.

Script: `ops/logit_n1_replay.py`, sha256 at writing
`a5ff4c7b515ea3c77223e63af9d12bbc6f341a5bb5b1086fa2239d2954559bcd`. If the script changes before the stamp, this line
is updated before the stamp. It imports `ops/logit_snr.py`, sha256
`82ec71f69f46bc52e733bd651d3b67e2bc6e747cfeed5c4354cd54b06c9611ec`, unchanged since `a73d3c9`.

Cost: **$0.** No card, no model, no network, no job. Mac, one thread, `nice -n 10`. Runtime *estimated* in §7.

Phase: MODE RECHERCHE on model quality, mandate O12 of `docs/CERVEAU.md`, step R1.

## 1. The question

N1: on six LLVQ arms scored on 2,280 MMLU questions, the measured accuracy sits under what iid Gaussian noise of the
arm's own residual amplitude on the four f16 option logits predicts. Bare Tetra reads 53.49 against 58.53 ± 1.05
predicted (*computed*, `docs/mesures/logit-snr-4b-2026-09-12.txt` section 5). The ± is the sd over simulation draws.
No interval exists on the 5-point gap, and the replay never ran on the sealed, served 4B or on the full split
(`docs/mesures/recherche-qualite-r0-depot-2026-11-13.txt` sections 3a, 3d).

Q7's overturn condition reads "the same replay on the sealed 4B where the Gaussian prediction falls inside the
measured interval". This prereg defines that interval: a paired, stratified interval on the per-question gap between
the arm's 0/1 outcome and the reference's predicted probability.

The question: on the sealed 4B, full split, does the measured accuracy sit under the Gaussian prediction, with a 95 %
interval that excludes zero?

## 2. The objects and the inputs

Every input is a per-question dump already committed: one row a question, the four answer-token logits verbatim
(`mmlu.rs` line 428). Full split: 14,042 questions, 57 subjects, census plan, fingerprint `a74a6d6213602979`, dtype
f16, dense reconstruction, kv f16, no `LLVQ_CONFIG`. On a census dump the stratified micro equals correct / 14,042.

Blob hashes from `git ls-files -s`. Rows and correct counts counted on 2026-11-13 by `grep`/`awk` on the raw lines,
to identify each dump against its journal; no statistic of this prereg was computed on them.

| arm | dump | blob | rows | correct | model line | journal, expected micro |
|---|---|---|---|---|---|---|
| sealed 4B | `docs/data/mmlu-dumps/mmlu-4b-abl-control-FULL.csv` | `e80ffe77` | 14,042 | 8,899 | `qwen3-4b-sealed.bin [LLVQ 2-bit, sealed]` | ablation-qualite-4b line 84, 63.3742 |
| FP16 4B | `docs/data/mmlu-dumps/mmlu-4b-f16-FULL.csv` | `8bd81bde` | 14,042 | 9,849 | `Qwen/Qwen3-4B [reference checkpoint]` | f16-full-2026-09-18 line 14, 70.14 |
| bare Tetra | `docs/data/mmlu-dumps/mmlu-4b-tetra-FULL.csv` | `8ff4ea17` | 14,042 | 7,673 | `qwen3-4b-tetra.bin [LLVQ 2-bit, sealed]` | tetranu-rowscales line 20, 54.64 |
| AWQ 4B | `docs/mesures/paper-table-2026-09-25-brut/paper-awq4b-census-2026-09-25/mmlu-4b-awq-FULL.csv` | `efbc5cd9` | 14,042 | 9,568 | `Pier-Jean/qwen3-4b-awq-deq [reference checkpoint]` | paper-table-2026-09-25 line 48, 68.14 |
| abl-qkv | `docs/data/mmlu-dumps/mmlu-4b-abl-qkv-FULL.csv` | `5920bba9` | 14,042 | 8,976 | sealed; restored q,k,v at f16, 108 matrices | ablation line 85, 63.9225 |
| abl-o | `…/mmlu-4b-abl-o-FULL.csv` | `289e4af5` | 14,042 | 8,881 | sealed; restored o, 36 | ablation line 86, 63.2460 |
| abl-gateup | `…/mmlu-4b-abl-gateup-FULL.csv` | `b1ee59d2` | 14,042 | 9,374 | sealed; restored gate,up, 72 | ablation line 87, 66.7569 |
| abl-down | `…/mmlu-4b-abl-down-FULL.csv` | `d792bd7c` | 14,042 | 9,181 | sealed; restored down, 36 | ablation line 88, 65.3824 |
| abl-attn | `…/mmlu-4b-abl-attn-FULL.csv` | `f5ec50c9` | 14,042 | 9,001 | sealed; restored q,k,v,o, 144 | ablation line 89, 64.1006 |
| abl-mlp | `…/mmlu-4b-abl-mlp-FULL.csv` | `a1d5413a` | 14,042 | 9,581 | sealed; restored gate,up,down, 108 | ablation line 90, 68.2310 |
| abl-all | `…/mmlu-4b-abl-all-FULL.csv` | `2fe2bb49` | 14,042 | 9,843 | sealed; restored all seven, 252 | ablation line 91, 70.0969 |
| sealed 8B | `docs/mesures/sealed-8b-27-2026-09-24-brut/mmlu-B/mmlu-8b-B-sealed-FULL.csv` | `8be4e759` | 14,042 | 9,770 | `qwen3-8b-sealed-B.bin [LLVQ 2-bit, sealed]` | sealed-8b-27 line 25, 69.58 |
| FP16 8B | `docs/data/mmlu-dumps/mmlu-8b-f16-FULL.csv` | `3ac768b9` | 14,042 | 10,539 | `Qwen/Qwen3-8B [reference checkpoint]` | census-8b line 24, 75.05 |
| sealed 14B | `docs/mesures/sealed-8b-14b-2026-09-23-brut/mmlu-14b/mmlu-14b-sealed-FULL.csv` | `30e17a0a` | 14,042 | 10,624 | `qwen3-14b-sealed.bin [LLVQ 2-bit, sealed]` | sealed-8b-14b line 24, 75.66 |
| FP16 14B | `docs/data/mmlu-dumps/mmlu-14b-f16-FULL.csv` | `52aca007` | 14,042 | 11,076 | `Qwen/Qwen3-14B@40c06982 [reference checkpoint]` | census-14b-ref line 1, 78.88 |

Identification notes:

- Sealed 4B: the control arm of the ablation job, file sha256 `886391a8`, picks 14,042 of 14,042 equal to the published
  63.37 dump, data rows identical (ablation lines 61-63; here `cmp` differs at line 2 only, the model path).
- Sealed 8B: `docs/ETAT.md` line 33 names `qwen3-8b-sealed-B.bin` at 69.58. Its `files.sha256` reads `7bdb9a55`, the
  arm B object of sealed-8b-27; its `mmlupair-vs-f16.txt` reads B 69.58 against A 75.05, fingerprints identical.
- Sealed 14B: `docs/ETAT.md` line 34, 75.66; `files.sha256` `61db37fe`; `mmlupair-vs-f16.txt` reads 75.66 against 78.88.
- The three f16 dumps in `docs/data/mmlu-dumps/` are byte-identical (`cmp`) to the copies in `f16-full-2026-09-18-brut`,
  `census-8b-2026-09-21-brut` and `census-14b-ref-2026-09-22-brut`. The two pair files above read the f16 dumps from
  another checkout (`../llvq-commit-8b/`); the micro they print, 75.05 and 78.88, matches the counts here.

C3 reads the 2,280-question dumps of `logit_snr.py` section 5 (fingerprint `65dcd53655e8bfa5`, 2,280 rows each):
`mmlu-4b-f16.csv` `7b2e8693`, `mmlu-4b-awq.csv` `b3a94753`, `mmlu-4b-llvq.csv` `7e6f8b16`, `mmlu-4b-tetra.csv`
`29f7241a`, `mmlu-4b-tetra-q5.csv` `09c64fce` (all in `docs/data/mmlu-dumps/`), `docs/data/bruit-mmlu-graines/`
`mmlu-s1.csv` `46ab0093`, `mmlu-s2.csv` `142be777`, `mmlu-s3.csv` `7913df5a`.

## 3. The arms

Each arm is scored against its own size's f16 dump on K, the questions present in both. The script refuses an arm
whose keys, answers or subject populations differ from its reference's; no arm is in its `INTERSECT` list, so |K| =
14,042 for every arm, and the script prints it.

| arm | role | reference |
|---|---|---|
| sealed 4B | **primary**, the only arm a decision rule reads | FP16 4B |
| bare Tetra, AWQ 4B, seven ablation arms | secondary, descriptive | FP16 4B |
| sealed 8B | secondary, descriptive | FP16 8B |
| sealed 14B | secondary, descriptive | FP16 14B |

The ablation arms restore groups of projections to f16 inside the sealed file; `abl-all` is FP16 except the int4
embedding and tied head. AWQ is the sign check: section 5 read it over its prediction on 2,280 q (+0.83).

## 4. Controls. Rule D: if one fails, the run describes and decides nothing

| control | what | when |
|---|---|---|
| C1 | `uv run ops/logit_snr.py`, unchanged, reproduces section 5 of `logit-snr-4b-2026-09-12.txt` character for character | re-run before the stamp (§9), and again in `controls` |
| C2 | the measured micro the new script recomputes for every full-split dump of §2 equals the journal figure to 0.01 pp | `controls`, after the stamp |
| C3 | on the seven section-5 arms (2,280 q), each of the three predictions falls within 3 journal sd of the journal's value | `controls`, after the stamp |
| C4 | null synthetic arms, y = beta f16 + iid Gaussian: G_gauss's 95 % normal interval contains 0 in at least 17 of 20 | selftest before the stamp at beta 0.6, sd 1.3, on the 2,280-q f16 dump and on a census-shaped copy of it; `controls` after the stamp at the sealed arm's fitted beta and sd, on the real FP16 4B census dump |
| C5 | planted couplings resolved with the sign derived below | selftest, before the stamp |
| C6 | three mutants of the script, each caught by a unit check or C4 | before the stamp |

C5 arms, census-shaped reference (each subject at its full-split population, question j a copy of the j mod 40-th of
the 2,280-q sample), beta 0.6, noise sd 1.3 a logit before centring:

- (a) tail: the same total variance placed only on the options f16 ranks third and fourth.
- (b) margin: every option's sd proportional to (f16 top-two margin + 1), scaled so the mean variance is 1.3².
- (c) letter: iid noise plus a constant +1.0 on option B.
- (d) head, added beside the brief: the same total variance only on the options f16 ranks first and second. It is the
  only planted arm whose gap has N1's sign.

Signs derived before C5 ran, by the exact integral of P(argmax = answer) over each question (`selftest
--derive-only`, *computed*, neglecting the error of the fitted beta and sd):

| arm | expected accuracy | expected Gaussian prediction | expected G_gauss | free reshuffle | banded reshuffle |
|---|---|---|---|---|---|
| (a) tail | 67.18 | 65.72 | **+1.46** | positive: a donor's noise lands on random ranks | positive, same reason |
| (b) margin | 67.43 | 65.72 | **+1.70** | positive: magnitude leaves its margin | near 0: the band keeps magnitude with margin |
| (c) letter | 64.97 | 65.07 | −0.10, reported only | near 0: a permutation keeps the option index | near 0 |
| (d) head | 64.60 | 65.72 | **−1.13** | negative | negative |

Pass rule for C5: the 95 % normal interval of G_gauss lies above 0 for (a) and (b) and below 0 for (d). (c) has no
pass rule. Every planted gap is smaller than N1's 5 points on 2,280 q; an instrument that resolves these resolves
N1's size.

## 5. The statistic

For arm A, reference F, key set K, subject s with population weight w_s = pop_s / Σ pop and n_s questions in K:

- Fit (`logit_snr.fidelity`): centre each question's four logits on their mean, pool, regress y = beta x + e;
  sd = sd(e); SNR = beta sd(x) / sd.
- c_k = A's own `correct` for question k.
- Gaussian iid: p_k = share of D = 200 draws in which argmax(beta x_k + ε), ε ~ N(0, (sd √(4/3))²) iid on the four
  options, x_k the raw f16 logits, equals the answer. Ties go to the first index, as `max()`.
- Free reshuffle: p_k = share of D = 200 random permutations of the residual vectors e_j across all of K in which
  argmax(beta x̄_k + e_π(k)) equals the answer, x̄ centred (`logit_snr.replay`, bands = 1).
- Banded reshuffle: the same, the permutation inside each of 20 equal-count bands of f16 top-two margin (bands = 20).
- Gap: G_R = 100 Σ_s w_s mean_{k∈s}(c_k − p_k). SE_R = 100 √(Σ_s w_s² var_s(c − p) / n_s), sample variance
  (n_s − 1), no finite-population correction. 95 % interval G_R ± 1.96 SE_R.
- Cross-check: stratified bootstrap, 10,000 resamples of questions within subject, p_k held fixed, percentiles 2.5 and
  97.5. The three references of one arm share one resample stream.
- Printed with them: beta, sd, SNR, measured micro and its SE (`logit_snr.accuracy_se`), each predicted accuracy
  Σ_s w_s mean p_k and its Monte Carlo SE √(Σ_s w_s² Σ_k p_k(1 − p_k) / D) / n_s.
- Seeds: `SEED = 20261113`; each (arm, statistic) draws from `random.Random` seeded with the first 8 bytes of
  SHA-256(`"SEED|arm|statistic"`).

Limit: beta and sd are fitted on the same questions, and p_k is held fixed. The interval is conditional on that fit
and does not carry the fit's own uncertainty. Under the null it is calibrated at the census size (C4: sd of G over 20
arms 0.252 against a mean SE 0.265) and conservative on 2,280 q (0.704 against 1.006), *computed*, §9.

## 6. Decision rules, written before the data

- **P1 (primary).** N1 **HOLDS** on the sealed 4B if the 95 % normal interval of G_gauss lies entirely below 0. N1
  **FALLS** on the sealed 4B if that interval contains 0 or lies above it; then Q7's overturn condition fires, N1
  keeps its older-arm scope only, and the phase reports to the operator before any further step.
- **P2.** If N1 holds: the size of the phase's gap is G_gauss with its interval, and the share the f16 margin does not
  carry is G_banded with its interval. No other reading is drawn from the secondary arms than their own G and
  interval; they are descriptive, and each still has a signed prediction.
- **Rule D.** If C1, C2, C3 or C4 fails in `controls`, or C4, C5 or a unit check fails in the selftest, the run
  describes and decides nothing. `controls` runs before `run`; both outputs go into the journal whole.
- The bootstrap interval is a cross-check. If it and the normal interval disagree on the side of 0 for the primary
  arm, the disagreement is reported and P1 still reads the normal interval.

## 7. Cost

- $0: dumps in the repository, pure Python, no dependency, no network.
- Runtime on the Mac, *estimated* from one census-sized synthetic arm timed on 2026-11-13: Gaussian p_k 2.8 s, each
  reshuffle 1.2 s, bootstrap 11.5 s, about 17 s an arm. `run` on the twelve arms about 3.5 min; `controls` about
  1.5 min (C1 7 s, C4 twenty Gaussian arms at 2.8 s). The full selftest took 145 s.

## 8. What does not compare

- 2,280-q figures against full-split figures: other questions, other sampling plan.
- G across sizes as a law. Three sizes are not a law.
- A G whose interval contains 0 is "not resolved", never "zero".
- G is a gap to a model of the noise on four logits. It says nothing about which weights or activations carry the
  error, and nothing about GSM8K, perplexity or the kernel path.
- The ablation arms share the sealed file and the same questions; their G values are not independent, and no
  difference between two of them is read here.

## 9. Dated disclosure, and the signed predictions

Run before the stamp, all on 2026-11-13, all *computed*, no full-split dump read by a statistic of this prereg and no
real quantized arm scored by the new script:

- C1 re-run: `uv run ops/logit_snr.py` at `44e62aa`. Its whole output equals the journal's 176 lines character for
  character, section 5 included. 6.8 s.
- `selftest --derive-only`: the C5 table of §4.
- One full `selftest` started and stopped about a minute in, before it printed anything (buffered output), so that
  this file could state the C5 signs before any C5 arm ran.
- `selftest --quick` (17 s) and `selftest` (145 s), synthetic arms built from `mmlu-4b-f16.csv` (2,280 q) only. Every
  check passes:
  - U1, the fit reads back beta 0.5984 for 0.6 and a simulation sd 1.3017 for 1.3.
  - U2, Monte Carlo p_k against the exact integral on 300 questions at D = 2,000: rms z 0.994, mean difference
    −0.004 pp.
  - U3, normal SE 1.031 against a bootstrap 1.016; an arm missing one key is refused; streams fixed by name.
  - C4, 2,280 q: the Gaussian interval covers 0 in 20 of 20 (both reshuffles 20 of 20), mean G −0.038, sd 0.704, mean
    SE 1.006. Census-shaped: 19 of 20, mean G +0.075, sd 0.252, mean SE 0.265.
  - C5, census-shaped, G_gauss [95 % normal]; free; banded:

    | arm | measured | G_gauss | G_free | G_banded | verdict |
    |---|---|---|---|---|---|
    | (a) tail | 67.53 | +1.85 [+1.41, +2.29] | +1.87 [+1.44, +2.29] | +1.79 [+1.36, +2.21] | passes, derived +1.46 |
    | (b) margin | 67.29 | +1.55 [+1.07, +2.02] | +1.19 [+0.74, +1.64] | −0.17 [−0.60, +0.25] | passes, derived +1.70 |
    | (c) letter | 64.93 | −0.11 [−0.65, +0.43] | +0.01 [−0.52, +0.53] | +0.02 [−0.51, +0.54] | reported, derived −0.10 |
    | (d) head | 64.63 | −1.06 [−1.62, −0.49] | −1.05 [−1.62, −0.49] | −1.11 [−1.67, −0.54] | passes, derived −1.13 |

    The bootstrap intervals agree with the normal ones to 0.02 pp on every row. The banded reshuffle absorbs the margin
    coupling (b), as derived. No reference sees the letter offset (c): its residual sd absorbs it and the gap is
    −0.11. A planted gap of 1 to 2 pp is resolved with a half-width near 0.5 pp, so a gap of N1's size, 5 pp on
    2,280 q, would be resolved by a wide margin at the census size.
- C6: three mutants, each applied at exactly one site (counted before applying), each run through `selftest --quick`,
  the file restored and its sha256 checked equal after each:
  - M1, `gauss_sd` returns sd without √(4/3): U1 fails (simulation sd 1.1273 for 1.3). C4 on 2,280 q does **not**
    catch it (19 of 20, mean G −1.151 against a mean SE 1.009). The full selftest, run once more with M1, catches it
    three ways: U1, census-shaped C4 (1 of 20, mean G −1.042) and C5 (b) (interval [−0.03, +0.90]).
  - M2, the Gaussian p_k reads the answer shifted by one letter: U2 fails (rms z 860), C4 fails (0 of 20, mean G
    +54.6).
  - M3, arm and reference swapped at the entry of `score`: C4 fails (0 of 20 on all three references, mean G +9.9).
  `git status` after the restores: the two new files only; `ops/logit_snr.py` sha256 unchanged.
- Read only: the headers, row counts and correct counts of the dumps of §2, by `grep`/`awk`.

Known before the stamp: N1's section-5 figures (§1); the full-split micro of every dump (§2); the ablation gains
(`docs/mesures/ablation-qualite-4b-2026-11-12.txt`); the 2,280-q fits of section 1 and 10 of the logit-snr journal
(sealed files never fitted).

### Signed predictions (main session)

Signed by the main session on 2026-11-13, before the stamp and before any statistic of this prereg read a
full-split dump. G in pp of micro accuracy, measured minus predicted, on 14,042 questions. The central values were
fixed and told to the operator before stage 1 of the instrument handed back, from one reading only: across the seven
section-5 arms the gap shrinks as accuracy rises (−6.55 at 52.19, −2.64 at 58.02, +0.83 for AWQ at 70.04, 2,280 q).
The AWQ row was set from that 2,280-q sample dump, before its full-split micro of 68.14 was read; it is kept as told.

P1 predicted: **HOLDS**, with a confidence of about 60 %. The case against: trained row scales distil the output
distribution, and may already have removed the structured share, leaving a G not resolved from 0.

| arm | G_gauss predicted | prediction interval | G_banded predicted | P1 outcome |
|---|---|---|---|---|
| sealed 4B (primary) | −1.2 | [−2.4, −0.2] | −0.8, interval [−1.8, 0.0] | HOLDS |
| bare Tetra | −4.8 | [−5.8, −3.8] | not predicted | n/a |
| AWQ 4B | +0.6 | [−0.2, +1.4] | not predicted | n/a |
| abl-qkv | −1.1 | [−2.2, −0.1] | not predicted | n/a |
| abl-o | −1.2 | [−2.4, −0.2] | not predicted | n/a |
| abl-gateup | −0.6 | [−1.5, +0.2] | not predicted | n/a |
| abl-down | −0.8 | [−1.8, +0.1] | not predicted | n/a |
| abl-attn | −1.0 | [−2.1, 0.0] | not predicted | n/a |
| abl-mlp | −0.3 | [−1.0, +0.4] | not predicted | n/a |
| abl-all | 0.0 | [−0.4, +0.4] | not predicted | n/a |
| sealed 8B | −0.9 | [−2.0, 0.0] | not predicted | n/a |
| sealed 14B | −0.5 | [−1.4, +0.3] | not predicted | n/a |

A prediction is "in" when the measured G falls inside its prediction interval. The count of predictions in and out
goes into the journal.
