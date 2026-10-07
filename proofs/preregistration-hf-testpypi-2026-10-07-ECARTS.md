# Deviations from `preregistration-hf-testpypi-2026-10-07.md`

The prereg is timestamped (`.ots` beside it, sha256 `b6f9c8a6`) and is not edited. Its deviations are written here,
each before the measurement it changes.

## É1. A second arm with transformers 5.19.0, added before any load

**Fact.** Control 3, the dry run of `pip install llvq-tetra` in an empty environment, resolved `transformers==5.19.0`.
That version was uploaded to PyPI on 2026-10-06 at 16:38 UTC, and the prereg did not see it: it pins the check to
5.18.0, the last version measured. A stranger who installs the package today gets 5.19.0, which nothing here has
loaded.

**Why it matters before the upload.** The metadata of 0.1.0 is fixed once it is on PyPI. Whether 5.19.0 works decides
whether 0.1.0 needs an upper bound on `transformers`.

**Change.** The check of section 3 runs twice, in two empty environments that differ by one pin:

| arm | torch | transformers | role |
|---|---|---|---|
| A | 2.14.1 | 5.18.0 | the prereg as written; its decision rule applies unchanged |
| B | 2.14.1 | 5.19.0 | what `pip install llvq-tetra` gives on 2026-10-07 |

Arm B reuses the Hub cache arm A filled. The unauthenticated download from an empty cache is arm A's control 6.

**Rule for arm B.** 256 ids of 256 and no missing or unexpected key: no upper bound. Anything else: the PyPI upload
waits, and an upper bound `transformers<5.19` or a fix goes to the operator.

Written on 2026-10-07, after the upload to TestPyPI and controls 1 to 3, before the first load of either arm.
