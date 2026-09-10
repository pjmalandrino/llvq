# Écarts au pré-enregistrement F1e §0 du 2026-09-10

Le préreg est tamponné et ne s'édite pas. Ce qui le corrige est ici.

## É1 — Le premier essai a refusé, comme Q5 l'annonçait. 0,02 $.

Job `6aa222f15527934177ebf189`, `l40sx1`, 46 s facturées.

  `oracle` : **MATCH** — la passe avant observable reproduit celle de candle
  sur cuda. Règle dure 10 satisfaite ; le harnais est celui des autres runs.

  `fusedrun` : `Error: tetra48: ClassTable: no runtime layout for Tetra`

**Q5 prédisait « quelque chose refuse au premier essai, 60 % ». Juste**, et
c'est tout l'intérêt d'avoir mis 0,40 $ avant 8 $.

**La cause.** `Transcoder::for_kind` construisait une `ClassTable`
**inconditionnellement**, avant de lire un seul bloc. Cette table décrit les
383 classes de la balle v1 ; un mot Tetra n'en nomme aucune, donc
`ClassTable::for_kind` la refuse — correctement. Les portes par `kind` de
l'étape 6.6 étaient ouvertes ; celle-ci est **en amont** d'elles.

**Pourquoi le Mac ne l'avait pas vu.** Le test portable
`a_tetra_file_has_no_runtime_layout_before_f1d` balayait les quatre
dispositions de balle et s'arrêtait là. **La disposition que toute l'étape 6
existe pour servir était la seule que le test ne construisait jamais.** Un
balayage qui omet son sujet n'est pas un balayage. `Tetra48` y est maintenant
un cas POSITIF.

**La correction.** `table: Option<ClassTable>`, bâtie pour les dispositions qui
lisent une classe. Le `Some` est l'autorisation — le motif de `searcher` et de
`g70`. Un fichier Tetra sous `planes14` meurt toujours là, par son nom.

## É2 — Trois autres tests rouges, trouvés en tirant le fil

Aucun n'est de cette nuit. Tous épinglaient un contrat qu'une étape avait
délibérément changé, et **tous étaient rouges depuis, sans être lus** :

| test | contrat qu'il épinglait | étape qui l'a changé |
|---|---|---|
| `the_gpu_bins_refuse_a_mixed_file_off_a_card` | un fichier mixte est refusé par toute disposition | 6.6 : int4 est orthogonal à la disposition |
| `rtbits_refuses_a_mixed_file_by_name` | `rtbits` refuse un fichier mixte | 6.7 : il le facture |
| `rtbits_refuses_a_tetra_file_by_name` | `rtbits` refuse un fichier Tetra | 6.7, même raison |

**Le premier m'a échappé plus tôt dans la nuit** parce que j'avais tronqué la
sortie de `cargo test` à `head -15` et que l'échec était sous la coupe. J'ai
écrit « (clean = ok) » sur une suite rouge. La parade : compter les
`test result:` au lieu d'en lire les quinze premières lignes.

## É3 — Un refus qui nommait une étape

« no runtime layout for Tetra **before F1d** », épinglé en huit endroits sur
cinq crates. F1d a tourné ce matin. Le refus, lui, n'a pas bougé d'un pouce —
cette table lit des classes de balle et le lira toujours — mais la phrase se
lisait « ça marchera une fois F1d passé ». Réécrite partout.

## É4 — Le coût

0,02 $ pour ce job. Cumul wave 3 : **3,50 $** sur 20,00 $.
