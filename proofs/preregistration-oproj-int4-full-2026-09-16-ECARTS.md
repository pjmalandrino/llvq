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
