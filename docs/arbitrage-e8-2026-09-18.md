# E8 arbitration: a simplicity probe, not a quality hunt

The decision is whether to fund an E8 comparison for step 4 of the Tetra diagnostic. The answer
this dossier prepares: fund it, but for the right reason. Lattice theory bounds what E8 can
lose to Leech at **9.0 % of MSE**, which is smaller than any lever the record has measured. So
E8 cannot explain the 4.3 pp that separates us from the paper's LLVQ. What it can do, if it
ties, is replace 7,138 lines of Λ₂₄ machinery with a 455 kB table.

Cost of this dossier: $0, no run. Arithmetic over numbers already in the repository.

## 1. The paper's E8 number does not answer the question

`llvq-paper-notes.md` Table 3 gives, at 2 bits per dimension on a Gaussian source:

| method | dim | MSE | SQNR bits | retention |
|---|---|---|---|---|
| E8 (cubic) | 8 | 0.103 | 1.64 | 82.0 % |
| LLVQ/Leech, shape-gain | 24 | 0.078 | 1.84 | 92.1 % |

That is 10.1 points of retention, and it is **not a lattice comparison**. The E8 row is cubic
shaping; the Leech row is shape-gain with a gain bit. The two differ in lattice and in shaping
at once, and the paper never measures E8 with shape-gain. Citing 10.1 points as the cost of
leaving Λ₂₄ would be reading a confound.

## 2. What the lattice part alone is worth

The normalized second moment of a product lattice equals that of its factor, so E8³ over a
24-dimensional block carries G(E8).

  G(E8) = 0.071682 · G(Λ₂₄) = 0.065771 · ratio 1.0899 (*cited*, Conway and Sloane, SPLAG
  Table 2.3)

Leech is therefore ahead by **9.0 % of MSE, 0.374 dB**, on the lattice alone.

Caveat, and it is the same one row 3 of `ROADMAP-QUALITY` carries: those two constants are
**not transcribed in this repository**, so the citation is unsourced until they are. Verifying
them is stage 0 below, and it is cheap.

## 3. The signed prediction

Applying the 9.0 % to the paper's own measured Leech shape-gain point (*computed*):

| arm | MSE | SQNR bits | retention |
|---|---|---|---|
| Leech shape-gain, 1 gain bit (*measured* by the paper) | 0.078 | 1.843 | 92.14 % |
| E8³ shape-gain, 1 gain bit (*predicted* here) | 0.085 | 1.778 | 88.9 % |

**3.2 points of retention, against the 10.1 the paper's table displays.** That is the number to
hold the experiment to.

One reading of that prediction decides more than the prediction itself. The paper measures
`norm(Λ₂₄(13))` with **zero** gain bits at MSE 0.085, and adding one gain bit takes it to
0.078: **8.2 % of MSE for one bit**. The lattice upgrade from E8 to Λ₂₄ is worth 9.0 %. The two
are the same order, so the representation's power sits in the shape-gain scheme at least as
much as in the lattice. If that survives measurement, the lattice is not where the quality is.

## 4. The rate matches, almost for free

E8's theta series is 240·σ₃(n) vectors at norm 2n, so a norm-capped codebook enumerates
exactly (*computed*):

| norm cap | points | bits per 8 dims | integer index | 24-dim block | b/weight | with 1 gain bit |
|---|---|---|---|---|---|---|
| 8 | 26,640 | 14.701 | 15 | 45 bits | 1.8750 | 1.9167 |
| 10 | 56,880 | 15.796 | 16 | 48 bits | 2.0000 | 2.0417 |
| 12 | 117,360 | 16.841 | 17 | 51 bits | 2.1250 | 2.1667 |

Tetra's measured stream is **1.9907 b/weight**
([references-comptabilite-2026-09-18](mesures/references-comptabilite-2026-09-18.txt)). The
norm-10 cap lands 0.5 % above it without a gain bit and 2.6 % above it with one. A cap between
shells 10 and 12 matches it exactly, which is the same device the Λ₂₄ side already uses with
its m ≤ 13 class cap. Rate matching is not the obstacle.

## 5. What it costs to build, and why it is small

E8 is small enough to brute-force. The norm-10 codebook is 56,880 points of 8 coordinates,
**455 kB as i8**, which fits in cache. Exact nearest-neighbour over it is a table scan, so the
diagnostic needs no search engineering at all.

That is the whole asymmetry with Λ₂₄. The existing machinery is 7,138 lines (*measured*,
`llvq-core/src` 572, `llvq-search/src` 4,373, `llvq-search/src/tetra` 2,193), and it exists
because 196,560 points in 24 dimensions under a 2 b/dim budget demanded a trellis, Golay
classes, parity repair and a ranking. None of that has an E8 analogue that needs writing.

`llvq-search/src/generic.rs` is **not** generic over lattices despite its name: it is generic
over Λ₂₄ classes, and it reads Golay codeword membership. E8 is new code, in `llvq-bench`,
outside the five crates under `forbid(unsafe_code)` and touching no format.

Verification, mirroring `leech_kissing_number_196560`: kissing number 240, the theta
coefficients 240, 2160, 6720, 17520, 30240, minimum norm 2, determinant 1, and membership of
every enumerated point (all-integer or all-half-integer, even coordinate sum). Those are exact
tests, not tolerances.

## 6. The three options

| option | what it buys | cost |
|---|---|---|
| **A. Fund stage 0 then 1** | a matched E8³ against Tetra on the v64 dump set, and the two cited constants verified | ~1 day, ~300 lines in `llvq-bench`, 5 exact tests, $0 of card |
| B. Skip E8 | step 4 reduces to Tetra against multi-shell Leech, which exists (`leech0c13`, `leech1c12` measured) | $0, and the diagnostic loses its external reference |
| C. Cite the paper's table | nothing. The number is confounded, section 1 | $0 |

Stage 0 is the E8 decoder and its invariants, plus a Monte-Carlo G(E8) against the cited
0.071682. Stage 1 is the matched codebook scored on the 30 cells of
[tetra-diag-4b-2026-09-18](mesures/tetra-diag-4b-2026-09-18.txt), reading the v64 arm, on
J_local and on reconstruction error.

Stage 1 gets a stamped prereg before it runs, with the 3.2 points of section 3 as its
registered prediction.

## 7. What follows from a tie, which is the likely outcome

If E8³ lands within a few percent of Tetra, the question stops being about quality. A 455 kB
table with a table-scan decoder against a trellis and 7,138 lines is a kernel and maintenance
argument, and it is worth having on its own terms: `tv_tetra48_h` carries no segmented kernel,
which is why `LLVQ_FUSE` is 0 on the served object.

If E8³ loses by much more than 3.2 points of retention, the surplus is not the lattice, and
the diagnostic has found something the NSM does not explain. That outcome is more interesting
than a win.

## 8. What this dossier does not establish

Nothing is measured here. The 9.0 % is cited and untranscribed, the 3.2 points are computed
from it, and no E8 point has ever been encoded in this repository. The b/weight column assumes
an integer index per E8 block and ignores what a real packing would pay in addressing.
