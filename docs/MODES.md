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

**Opened by** the operator, with a mandate. The mandate is half a page and fixes four things, no more:

| field | content |
|---|---|
| Axis | One fundamental criterion to move: quality, decode speed, time to first token, memory, or the cost of producing an object. The others become constraints for the phase. |
| Starting point | A gap to a bound, or an anomaly: a measured fact in `CERVEAU.md` that nothing explains. One is enough. |
| Untouchable | What the phase may not change, named one by one. Everything not named is free: the format, bit identity, the dimension of the code, the kernel. |
| Budget | Days, dollars, and whether the card is allowed. The phase closes when it runs out, whatever it holds. |

The mandate does not name the mechanism, the search space or the expected gain. Those are the phase's work. An idea
brought from outside, a paper or a hunch, enters here as a starting point and goes through the phase like any
other.

A phase follows one of two routes, chosen at R2 by what the axis allows.

- **The search route**, when a verifier exists: candidates in number, scored automatically. It is the route of
  program search under an evaluator. It finds what a framing contains; the framing itself, R1 to R3, is where the
  phase is won or lost.
- **The explanation route**, when no fast verifier exists: an anomaly, several rival hypotheses that explain it,
  and for each the cheapest experiment whose result would kill it. The idea is the hypothesis that survives, and
  its prediction must be one that would surprise before the experiment. It is the route of strong inference.

| step | what is done | what must exist before the next step |
|---|---|---|
| R0. Read | The claims of the axis in `CERVEAU.md`, its open questions, its anomalies, its dropped beliefs. Then the literature: who has tried this, and what they got. | A list of what is already known, already refuted, and already published. |
| R1. Frame | One closed problem, one number nobody can argue with, and the bound on it or, failing a bound, the best published figure. | The gap between our figure and that reference. No gap, no phase. |
| R2. Verifier | Search route: an evaluation that is exact, automatic and cheap, seconds to minutes, on the Mac when possible. Explanation route: the killing experiment of each hypothesis, with its cost. | Search: the verifier ranks known objects as the journals do, a mutant of it is caught, and what it scores is the axis itself or a quantity `CERVEAU.md` holds as faithful to it. Explanation: each hypothesis predicts a different result. |
| R3. Space | Search: the candidates as formal objects, programs, constructions, parameters. Explanation: at least three rival hypotheses, one of them the dull one. | Search: the space is wider than any space where a ceiling is already recorded. Explanation: no hypothesis is safe from every experiment. |
| R4. Loop | Search: agents generate candidates in number, the verifier scores them, the best are kept and mutated. Explanation: the experiments run cheapest first, each under a stamped prereg and the operator's go when it costs. | Every candidate or experiment and its result is logged. Nothing is judged by argument. |
| R5. Read the winners | The surprising winners, or the surviving hypothesis, are studied and the principle is written down. | A conjecture, entered in `CERVEAU.md` with standing `conjecture`. |
| R6. Break it | Independent agents attack the principle; held-out data, exhaustive enumeration or a proof where one exists. | The principle survives, or it is recorded as refuted. |

**Closed by** one of three outcomes, each written into `CERVEAU.md` in the closing commit. A ceiling or a killed
hypothesis is a result: it removes a place to look.

- **An idea.** A candidate beats the best known under the verifier, beyond noise, on data the loop never saw; its
  principle is stated; it survived R6. It leaves with an idea sheet.
- **A ceiling, or a dead hypothesis.** The space is exhausted, or the experiments killed every rival. What was
  excluded and the space it holds for are recorded, so the next phase looks elsewhere.
- **Nothing to stand on.** No verifier could be built and no hypothesis could be killed at the budget. The axis
  keeps its anomaly in `CERVEAU.md`, with what was tried.

**Forbidden in this mode:**

- Estimating a gain instead of scoring or measuring it.
- A paid run at full scale to see what happens.
- Keeping a candidate because it is elegant, or dropping one because it is strange.
- Reporting a per-model tweak as a result.

**The idea sheet** is one page: the principle in three sentences, the verifier and the score against the best
known, what R6 tried and failed to break, which fundamental criteria it should move and which it must not, and the
first full-scale measurement that would refute it.

### Which route each axis allows today

| axis | route | standing on 2026-11-12 |
|---|---|---|
| Memory: the code on Λ₂₄ | Search. Rate at equal distortion on real weight blocks, exact, seconds on the Mac; decoder cost in instructions. | Ready. The ceiling of 24-dimension fixed-rate codes, 0.60 dB under Tetra (*computed*), is recorded; the space must be wider. Distortion is faithful to memory at equal distortion. It is not faithful to quality: under about 0.4 dB it does not predict the sign of MMLU. |
| Speed: the kernel | Search, costly. Bit identity or the served tolerance, and time on the card. | A job must score a hundred variants, not one. |
| Quality of the model | Explanation. A full MMLU arm takes 22 minutes, and perplexity and KL predict it poorly. | Open through its anomalies: the error costs more than Gaussian noise of its size, and attention is worth more once the MLP is restored. A fast, faithful verifier would move it to the search route. |

## MODE DEVELOPPEMENT

Purpose: take one idea sheet to a gain measured on every fundamental criterion, on the served object, and seal it.
Nothing is searched for in this mode. If the idea fails, the phase closes and the failure goes to `CERVEAU.md`.

**Opened by** the operator, on one idea sheet from a closed research phase. Nothing else opens it: no hunch, no
paper, no measurement. Those go to MODE RECHERCHE as a starting point.

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

## The lab around the two modes

The project is a one-man lab. The roles are fixed so that nobody judges their own work. The delegation rules are in
[`METHODE.md`](METHODE.md) §8.

| role | who | what |
|---|---|---|
| Direction | the operator | Chooses the axis, signs the mandate, opens and closes phases, decides every kill and every serve. |
| Lead | the main session | Frames, signs predictions before a measurement, reads results against them, keeps `CERVEAU.md`. |
| Hands | agents | Generate, build, measure, attack: Opus agents execute, Sonnet agents write the simple. An agent that proposed an idea does not attack it. |
| Memory | `CERVEAU.md` and the journals | What holds, and the record of how it came to hold. |

At most one research phase and one development phase run at a time: one card budget, one work machine. Between
phases the lead brings the operator the ranked open questions and anomalies of `CERVEAU.md`, and the operator picks
the next mandate.

## What this method is, and where it is weak

It joins three practices used elsewhere. Program search under an evaluator gives the search route. Strong
inference, rival hypotheses and crucial experiments, gives the explanation route. Preregistration and the paired
interval, already in [`METHODE.md`](METHODE.md), guard both. Autonomous research pipelines add two steps we keep: a
literature check before generating (R0), and an attacker that is not the author (R6). We do not keep their habit
of ranking ideas by a model's opinion: that is how the night of 2026-11-11 failed.

Its known weaknesses, to watch at every phase review:

- **The framing is not automated.** The loop finds what a space contains. Choosing the space is still judgement,
  and a poor framing gives a clean ceiling on the wrong problem.
- **A verifier can be exact and beside the point.** The loop optimizes what is scored. R2 therefore asks that the
  scored quantity be the axis or a proven stand-in for it.
- **The loop games its verifier.** Held-out data at R6 is the only defence, and it is spent once.
- **Cheap verifiers pull the lab toward what is easy to score.** The explanation route exists so that the axis that
  matters most is not dropped for want of one.
- **One reader.** The operator is the only outside check. Predictions signed before each measurement, and their
  score kept in the journals, are how the lead's judgement is audited.

## Switching

- One mode at a time on an axis. Two axes may be in different modes.
- A phase opens with a commit that enters its mandate or its idea sheet in `CERVEAU.md`'s open questions, and
  closes with a commit that records the outcome there.
- Research does not drift into development: an idea without its sheet is not developed. Development does not drift
  into research: a build that needs a new idea stops and says so.
- A measurement that only describes an object belongs to neither mode. It is run when a phase needs it, and its
  result goes to `CERVEAU.md` as a claim with its scope.
