# The two working modes

The project works in phases, and a phase is in one of two modes. MODE RECHERCHE finds a new idea. MODE DEVELOPPEMENT
turns one validated idea into a measured gain on the served object. The operator opens and closes each phase by
name. [`CERVEAU.md`](CERVEAU.md) is the memory both modes read first and write last. The lab rules of
[`METHODE.md`](METHODE.md) hold in both.

Operator, 2026-11-12. The night of 2026-11-11 searched for ideas by brainstorming and judged them by estimates; it
found none. This document replaces that way of searching.

## What counts as an idea

An idea is a new mechanism, with a principle that can be stated, that changes how every future quantization is
made or served. It moves a fundamental criterion: quality, speed, or memory.

These are not ideas, in either mode:

- A setting changed on one model: a projection type moved to int4, a different calibration draw, a tile size.
- A gain projected through a proxy, such as decibels of a codebook turned into MMLU points.
- A law drawn from a few points, such as three model sizes.
- A measurement that describes one object and proposes nothing.

## MODE RECHERCHE

Purpose: find an idea nobody has stated, by search under a verifier, then understand it. The idea is not imagined
and then tested. It is found among the winners of a search, and the principle is read from them afterwards.

**Opened by** the operator, with a front (quality, speed, memory) and a budget in days and dollars.

| step | what is done | what must exist before the next step |
|---|---|---|
| R0. Read | The claims of the front in `CERVEAU.md`, its open questions, its dropped beliefs. | A list of what is already known and already refuted. |
| R1. Frame | One closed problem, one number nobody can argue with, and the known bound on it. | The gap between the best known and the bound. No gap, no phase. |
| R2. Verifier | An evaluation that is exact, automatic and cheap: seconds to minutes, on the Mac when possible. | The verifier ranks known objects as the journals do, and a mutant of it is caught. |
| R3. Space | The candidates as formal objects: programs, constructions, parameters. | The space is wider than any space where a ceiling is already recorded. |
| R4. Loop | Agents generate candidates in number, the verifier scores them, the best are kept and mutated. | Every candidate and its score is logged. No candidate is judged by argument. |
| R5. Read the winners | The surprising winners are studied and the principle behind them is written down. | A conjecture, entered in `CERVEAU.md` with standing `conjecture`. |
| R6. Break it | Independent agents attack the principle; held-out data, exhaustive enumeration or a proof where one exists. | The principle survives, or it is recorded as refuted. |

**Closed by** one of three outcomes, each written into `CERVEAU.md` in the closing commit:

- **An idea.** A candidate beats the best known under the verifier, beyond noise, on data the loop never saw; its
  principle is stated; it survived R6. It leaves with an idea sheet.
- **A ceiling.** The space is exhausted. The ceiling and the space it holds for are recorded, so the next phase
  searches elsewhere.
- **No verifier.** R2 failed. The front is closed to research until a verifier exists, and building one is a phase
  of its own.

**Forbidden in this mode:**

- Estimating a gain instead of scoring it.
- A paid run at full scale to see what happens.
- Keeping a candidate because it is elegant, or dropping one because it is strange.
- Reporting a per-model tweak as a result.

**The idea sheet** is one page: the principle in three sentences, the verifier and the score against the best
known, what R6 tried and failed to break, which fundamental criteria it should move and which it must not, and the
first full-scale measurement that would refute it.

### Where a verifier exists today

| front | verifier | standing on 2026-11-12 |
|---|---|---|
| The code on Λ₂₄ (memory, quality at equal rate) | Distortion on real weight blocks, exact, seconds on the Mac; decoder cost in instructions. | Ready. The ceiling of 24-dimension fixed-rate codes, 0.60 dB under Tetra (*computed*), is recorded; the space must be wider. |
| The kernel (speed) | Bit identity or the served tolerance, and time on the card. | Costly. A job must score a hundred variants, not one. |
| Model quality | None. A full MMLU arm takes 22 minutes, and perplexity and KL predict it poorly. | Closed to research until a fast, faithful verifier exists. |

## MODE DEVELOPPEMENT

Purpose: take one idea sheet to a gain measured on every fundamental criterion, on the served object, and seal it.
Nothing is searched for in this mode. If the idea fails, the phase closes and the failure goes to `CERVEAU.md`.

**Opened by** the operator, on one idea sheet.

| step | what is done | gate |
|---|---|---|
| D0. Plan | The table of fundamental criteria against the served object: what must move, what must hold. Cost and days. | The operator's go. |
| D1. Build | The code, with its tests. Mutate before calling a gate green. | Fast loop and clippy clean; full suite if a format or the indexing is touched. |
| D2. Prereg | The question, the arms, the pass rule and the signed predictions, stamped before the first measurement. | The stamp, the cost announced, the operator's go. |
| D3. Measure | On the served object, every fundamental criterion, medians with round-by-round ranges. | The result is read against the prereg's rule, to the letter. |
| D4. Reproduce | The same measurement on a second size before any general claim. | A claim in `CERVEAU.md` says only what its scope covers. |
| D5. Seal | Journal, registry row, PR, independent review, merge. The same commit edits the claims the journal touches. | CI green; nothing served changes. |
| D6. Serve | A served config changes only on the operator's decision. | The operator. |

**The rule that never bends:** a criterion that improves does so with no loss on the others. Quality, whole-model
b/param, VRAM, decode speed, time to first token and the cost of producing the object are all read at D3.

**Closed by** a sealed gain, or by a refutation recorded in `CERVEAU.md` with the journal that carries it. A
refuted idea goes back to MODE RECHERCHE only as a new phase, on a new space.

## Switching

- One mode at a time on a front. Two fronts may be in different modes.
- A phase opens with a commit that names the mode, the front and the budget in `CERVEAU.md`'s open questions, and
  closes with a commit that records the outcome there.
- Research does not drift into development: an idea without its sheet is not developed. Development does not drift
  into research: a build that needs a new idea stops and says so.
- A measurement that only describes an object belongs to neither mode. It is run when a phase needs it, and its
  result goes to `CERVEAU.md` as a claim with its scope.
