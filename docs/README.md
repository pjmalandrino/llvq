# The repository documents

Read in this order:

| document | content |
|---|---|
| [`ETAT.md`](ETAT.md) | where things stand: served configuration, numbers, open decisions |
| [`ROADMAP.md`](ROADMAP.md) | what comes next, with gates and costs |
| [`how-it-works.md`](how-it-works.md) | the maths, the Leech lattice, the Tetra format, the fused kernel and 4E |
| [`METHODE.md`](METHODE.md) | the lab rules: prereg, labels, noise, tests |
| [`STYLE.md`](STYLE.md) | how these documents are written |
| [`HISTORIQUE.md`](HISTORIQUE.md) | the dated thread; append at the bottom, the past is not rewritten |

Plans still in use:

| document | content |
|---|---|
| [`plan-300.md`](plan-300.md) | the study of 4E's speed past v6, with its three measured steps |
| [`recherche-debit-maths.md`](recherche-debit-maths.md) | what the lattice's maths can buy in throughput, ranked, with its ceilings |
| [`plan-metal-debit.md`](plan-metal-debit.md) | the Mac: the MLX door, the Rust Metal path, what comes next |
| [`plan-transformers.md`](plan-transformers.md) | loading a Tetra file in `transformers`: stages, gates, upstream |
| [`plan-runtime.md`](plan-runtime.md) | the `llvq` runtime: design, stages, decisions |
| [`ROADMAP-QUALITY.md`](ROADMAP-QUALITY.md) | the quality leads of September 2026, cited by many journals |

Reference:

| document | content |
|---|---|
| [`format-noyau.md`](format-noyau.md) | the VRAM layouts, the bit accounting, the measurement traps |
| [`fiche-4b.md`](fiche-4b.md) | paper 1's `Planes14` 4B, number by number |
| [`llvq-paper-notes.md`](llvq-paper-notes.md) | the source paper, transcribed. Never reopen the PDF |
| [`qtip-provenance.md`](qtip-provenance.md) | where the bench's QTIP kernel comes from, and why it is not redistributed |
| [`hub/`](hub/) | the Hub cards as published. The three `-sealed` cards and the safetensors one do not move; `qwen3-4b-served.md` is the moving card |
| [`upstream/`](upstream/) | contributions to other projects: the `candle` issue, the `transformers` proposal |
| [`templates/`](templates/) | experiment, prereg, journal and deviation templates |

Frozen, never edited:

| directory | content |
|---|---|
| [`mesures/`](mesures/) | one journal per measurement, dated, with its raw output |
| [`data/`](data/) | the CSVs: jobs and costs, per-question dumps, figure data |
| [`archive/`](archive/) | dated documents: old plans, handovers, audits. Some claims there were later refuted |
| [`../proofs/`](../proofs/) | the timestamped preregs and their deviations |

A path `docs/<name>-<date>.md` cited in an older journal or prereg now resolves under `archive/`. Documents removed on
2026-11-07 (the HTML courses, the 27B plans, `plan-debit-noyau.md`, a few dated reports and 31 uncited archive
entries) are in git history.
