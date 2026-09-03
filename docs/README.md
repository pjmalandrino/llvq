# Les documents du dépôt

Par où reprendre, dans cet ordre :

| document | contenu | s'édite ? |
|---|---|---|
| [`ETAT.md`](ETAT.md) | où on en est : configuration servie, chiffres de tête, décisions ouvertes | oui, à chaque changement d'état |
| [`ROADMAP.md`](ROADMAP.md) | la suite : pistes, gates, coûts, décisions attendues | oui |
| [`HISTORIQUE.md`](HISTORIQUE.md) | le fil chronologique, une entrée par période | on ajoute en bas, on ne réécrit pas le passé |
| [`METHODE.md`](METHODE.md) | les règles du laboratoire : préreg, chiffres, bruit, tests, machines | oui, quand une règle change |
| [`STYLE.md`](STYLE.md) | comment on écrit ici | oui |
| [`templates/`](templates/) | gabarits : expérience, préreg, journal, écarts | oui |

Les références sur l'objet publié et le noyau, à jour mais longues :

| document | contenu |
|---|---|
| [`fiche-4b.md`](fiche-4b.md) | le Qwen3-4B publié, chiffre par chiffre, avec sa provenance |
| [`format-noyau.md`](format-noyau.md) | le format en VRAM, les layouts, les pièges de mesure |
| [`echelle-4b-8b-2026-08-08.md`](echelle-4b-8b-2026-08-08.md) | la courbe d'échelle 4B, 8B, 14B |
| [`campagne-finale-2026-08-07.md`](campagne-finale-2026-08-07.md) | la campagne de mesure du 4B |
| [`llvq-paper-notes.md`](llvq-paper-notes.md) | le papier source transcrit (cité par notre papier) |
| [`qtip-provenance.md`](qtip-provenance.md) | d'où vient le noyau QTIP du banc, et pourquoi il n'est pas redistribué |
| [`hf-model-card.md`](hf-model-card.md) | la carte du modèle sur Hugging Face |

Ce qui est figé et ne s'édite jamais :

| répertoire | contenu |
|---|---|
| [`mesures/`](mesures/) | un journal par mesure, daté, avec ses bruts |
| [`data/`](data/) | les CSV : jobs et coûts, dumps MMLU par question, données des figures |
| [`archive/`](archive/) | les documents d'époque : plans, passations, audits, brouillons. Ils peuvent contenir des affirmations démenties depuis |
| [`../proofs/`](../proofs/) | les préregs tamponnés et leurs écarts |

Le code se lit depuis [`../CLAUDE.md`](../CLAUDE.md) (carte des crates, commandes, variables).
