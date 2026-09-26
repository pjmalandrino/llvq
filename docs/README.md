# The repository documents

Where to resume, in this order. Each document stands on its own at its level.

| document | content | edited? |
|---|---|---|
| [`ETAT.md`](ETAT.md) | where things stand: served configuration, headline numbers, open decisions | yes, on every change of state |
| [`ROADMAP.md`](ROADMAP.md) | what comes next: gates, costs, decisions awaited | yes |
| [`ROADMAP-QUALITY.md`](ROADMAP-QUALITY.md) | the quality axis, ordered by feasibility | yes |
| [`HISTORIQUE.md`](HISTORIQUE.md) | the chronological thread, one entry per period | append at the bottom, the past is not rewritten |
| [`METHODE.md`](METHODE.md) | the lab rules: prereg, numbers, noise, tests, machines | yes, when a rule changes |
| [`STYLE.md`](STYLE.md) | how we write here, with the target length of each document | yes |
| [`templates/`](templates/) | templates: experiment, prereg, journal, deviations | yes |

Reference documents, up to date and long:

| document | content |
|---|---|
| [`fiche-4b.md`](fiche-4b.md) | the published `Planes14` Qwen3-4B, number by number, with its provenance |
| [`format-noyau.md`](format-noyau.md) | the VRAM layouts, the kernel, the measurement traps |
| [`modele-erreur.md`](modele-erreur.md) | the error model: what LLVQ optimizes, and what it should |
| [`inference-cost-reduction-2026.md`](inference-cost-reduction-2026.md) | survey of the field, and candidates to implement |
| [`llvq-paper-notes.md`](llvq-paper-notes.md) | the source paper, transcribed. Never reopen the PDF |
| [`qtip-provenance.md`](qtip-provenance.md) | where the bench's QTIP kernel comes from, and why it is not redistributed |
| [`hf-model-card.md`](hf-model-card.md) | the model card on Hugging Face. It describes the published `Planes14` file |

Teaching material, HTML, self-contained: [`cours-comprendre-llvq.html`](cours-comprendre-llvq.html),
[`cours-tetra.html`](cours-tetra.html), [`cours-layouts-runtime.html`](cours-layouts-runtime.html),
[`architecture-c4.html`](architecture-c4.html).

Frozen, never edited:

| directory | content |
|---|---|
| [`mesures/`](mesures/) | one journal per measurement, dated, with its raw output |
| [`data/`](data/) | the CSVs: jobs and costs, per-question MMLU dumps, figure data |
| [`archive/`](archive/) | period documents: plans, handovers, audits, drafts, and the dated working documents. They may contain claims that have since been refuted |
| [`../proofs/`](../proofs/) | the timestamped preregs and their deviations |

On 2026-09-26 every dated working document moved from `docs/` into [`archive/`](archive/), which left the fourteen
documents above. A path of the form `docs/<name>-<date>.md` cited in an older journal or preregistration, neither of
which is edited, now resolves under `docs/archive/`.

The second paper is in [`../paper2/`](../paper2/README.md). Read the code starting from [`../CLAUDE.md`](../CLAUDE.md),
which carries the crate map, the commands and the environment variables.
