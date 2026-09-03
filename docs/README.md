# The repository documents

Where to resume, in this order:

| document | content | edited? |
|---|---|---|
| [`ETAT.md`](ETAT.md) | where things stand: served configuration, headline numbers, open decisions | yes, on every change of state |
| [`ROADMAP.md`](ROADMAP.md) | what comes next: leads, gates, costs, decisions awaited | yes |
| [`HISTORIQUE.md`](HISTORIQUE.md) | the chronological thread, one entry per period | append at the bottom, the past is not rewritten |
| [`METHODE.md`](METHODE.md) | the lab rules: prereg, numbers, noise, tests, machines | yes, when a rule changes |
| [`STYLE.md`](STYLE.md) | how we write here | yes |
| [`templates/`](templates/) | templates: experiment, prereg, journal, deviations | yes |

Reference documents for the published object and the kernel, up to date but long:

| document | content |
|---|---|
| [`fiche-4b.md`](fiche-4b.md) | the published Qwen3-4B, number by number, with its provenance |
| [`format-noyau.md`](format-noyau.md) | the VRAM format, the layouts, the measurement pitfalls |
| [`echelle-4b-8b-2026-08-08.md`](echelle-4b-8b-2026-08-08.md) | the scaling curve, 4B, 8B, 14B |
| [`campagne-finale-2026-08-07.md`](campagne-finale-2026-08-07.md) | the 4B measurement campaign |
| [`llvq-paper-notes.md`](llvq-paper-notes.md) | the source paper, transcribed (cited by our paper) |
| [`qtip-provenance.md`](qtip-provenance.md) | where the bench's QTIP kernel comes from, and why it is not redistributed |
| [`hf-model-card.md`](hf-model-card.md) | the model card on Hugging Face |

Frozen, never edited:

| directory | content |
|---|---|
| [`mesures/`](mesures/) | one journal per measurement, dated, with its raw output |
| [`data/`](data/) | the CSVs: jobs and costs, per-question MMLU dumps, figure data |
| [`archive/`](archive/) | period documents: plans, handovers, audits, drafts. They may contain claims that have since been refuted |
| [`../proofs/`](../proofs/) | the timestamped preregs and their deviations |

Read the code starting from [`../CLAUDE.md`](../CLAUDE.md) (crate map, commands, variables).
