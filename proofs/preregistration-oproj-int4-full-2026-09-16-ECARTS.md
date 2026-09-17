# Deviations — preregistration-oproj-int4-full-2026-09-16

The prereg is timestamped and not edited. What it left open, or got wrong, is
recorded here, with the time it was decided relative to the result.

## E1. The interval's finite population correction was not specified

**Decided and committed BEFORE the paired result was computed** (git: this
commit precedes the journal that reports the number).

§3 asks for "the paired gain on the 11,762 questions that did NOT take part in
the selection, with its interval", and does not say whether the interval carries
a finite population correction. The two readings answer different questions and
differ a great deal in width here:

- **With FPC** (`mmlupair` default), each subject's deviation shrinks by
  `sqrt(1 - n/N)`. The held-out set is most of every subject's population — 1,494
  of 1,534 in `professional_law` — so the interval nearly closes. It answers
  "what is the difference on THIS exam", which a census knows almost exactly.
- **Without FPC** (`--no-fpc`), it answers "would the gain hold on other
  questions drawn like these" — the superpopulation question.

A confirmation exists to test whether a selection generalizes rather than
reflecting which questions happened to be in the selection set. That is the
second question. So:

**Primary: `--no-fpc`.** It is also the wider, more conservative interval, and it
is chosen without knowing which reading excludes zero. The FPC interval is
published beside it, labelled, and does not carry the decision.

## E2. Control 2 did not pass as written

**Recorded BEFORE the paired result was computed.**

§6 control 2 requires the reference arm to reproduce the exploration's shipped
arm on the 2,280 shared questions, "pick for pick". It does on **2,268 of
2,280**. Twelve picks differ (and six on the o_proj arm, informatively).

The cause is visible in the logits and is not a harness fault. The exploration
ran on `rtx-pro-6000` and the census on `l40sx1`, both in f16, and f16 kernels
differ across cards in the last bits. Every one of the six differing picks
inspected sits on an exact or near-exact tie — for example
`abstract_algebra,80`: logits 23.953125 / 23.609375 / **23.953125** / 23.3125 on
one card, where options A and C tie exactly, and 23.96875 / … / 23.953125 on the
other, where they no longer do. A tie broken differently is a different pick.

What it does and does not touch: it bears on whether the census is the same
object as the exploration, and the answer is "yes, up to tie-breaking across
cards". It does **not** enter the primary result, because the two arms being
compared both ran in one job, on one card, question for question. The control
was written too strictly for a cross-card rerun, and that is the prereg's
error, recorded here rather than waived.

## E3. What "did not take part in the selection" means, exactly

**Decided BEFORE the paired result was computed.**

A question is identified by `(subject, index)`, its position in the test split.
Verified: all 2,280 exploration rows carry the same text hash at the same
`(subject, index)` in the census. The exploration's sample is a seeded shuffle,
not the first 40 by index, so a positional rule would have been wrong.

MMLU's test split repeats some question texts — 27 hashes appear twice, 54
rows. Seven held-out rows share their text with a selection question:
`college_physics` 3, 27, 77, 97; `high_school_psychology` 396;
`professional_psychology` 478; `us_foreign_policy` 32.

- **Primary: 11,762 rows**, the complement of the 2,280 selection rows by
  `(subject, index)`. It is the population §3 names by count.
- **Sensitivity: 11,755 rows**, those seven removed, since their text was seen
  during selection. Published beside the primary. If the two disagree in
  conclusion, that is reported, and the stricter one is the one to believe.

Subset dumps carry a fingerprint derived from the census's (`sha256` of the
census fingerprint plus the subset name, first 16 hex digits), identical on both
arms, so no file claims to be the run it was cut from.

## E4. The budget claim of §1 compares two different units

**Found after the result, while checking the decision row's "within budget".**
It does not move the quality result; it changes why the budget holds.

§1 says the +0.1954 b/param surcharge "keeps the model under `b_max` = 3.00
(2.7645 → 2.9599)". Those two figures are **whole-model b/param**, embedding
included (hard rule 6). `b_max` = 3.00 is **kernel b/weight** on the projection
weights (`docs/ETAT.md:533`, the triplet's 27.93 GB left to the weights). The
exploration journal made the same comparison. It is a unit mismatch, and it
happened to give the right answer for the wrong reason.

Redone in kernel b/weight, over the 3,633,315,840 projection weights, with
Tetra at 2.1498 and int4 g128 at 4.250 (*computed*):

| how the int4 matrices are held in VRAM | shipped Q5 | + `o_proj` int4 | against 3.00 |
|---|---|---|---|
| served by a native int4 kernel | 2.2044 | **2.4226** | under, 0.58 of margin |
| dequantized to f16 | 2.5095 | **3.9485** | **over** |

Which one applies today: the served Q5 measures 0.97 GB of projections on card
(`docs/mesures/f1e0-2026-09-10.txt`), against 1.001 GB computed for the native
path and 1.140 GB for the dequantized one. `v_proj` is served natively.

So `o_proj` at int4 fits **on condition that it too is served by the native int4
kernel**. That kernel serves `v_proj`, 1024 × 2560; `o_proj` is 2560 × 4096, and
the kernel has not been verified on that shape. Until it is, "within budget" is
conditional, and the adoption decision has to know it.
