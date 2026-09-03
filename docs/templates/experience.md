# <Short name of the experiment> (<YYYY-MM-DD>)

**Question.** <One sentence.>

**Answer.** <One sentence, with the number and its interval.>

## Setup

- Object: <the file, the model, the size>
- Single variable: <what changes between the arms>
- Controls: <what does not change, and where they have already been measured>
- Cost: <measured, in $ or in machine hours>; duration <measured>

## Result

| arm | value | interval | source |
|---|---|---|---|
| control | | | |
| <arm 1> | | | |

## Controls

| control | expected | obtained | verdict |
|---|---|---|---|
| | | | passes / fails |

If a control fails, nothing gets published and the "Answer" line says so.

## What this does not establish

- <one caveat per line, the heaviest first>

## Decision

<What the result opens or closes, by the rule laid down in advance. If the rule
does not cover the case, say so, and name who decides.>

## Provenance

- prereg: `proofs/<file>.md`, sha256 `<first 8>`, timestamped on <date>
- deviations: `proofs/<file>-ECARTS.md` (or "none")
- code: commit `<7 chars>`
- job: `<id>`, <card>, <minutes>, <$>
- raw: `docs/mesures/<file>.txt` and `docs/mesures/<file>-brut/`
- data: `docs/data/<...>`
