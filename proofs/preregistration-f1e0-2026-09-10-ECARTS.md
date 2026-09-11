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

## É5 — Deuxième et troisième essais : deux noms, deux centimes chacun

**Essai 2** (`6aa231b021047bf1b03717c0`, 65 s, 0,03 $) :

  `Error: no kernel tv_planes_seg_h: named symbol not found`

Le rapport de registres était conditionné par `planes.is_some()`, c'est-à-dire
« la disposition n'est pas Slot32 ». Sur `Tetra48` c'est vrai et inutile : sa
liste de sources ne partage rien avec les dispositions de balle et ne porte pas
`tv_planes_seg_h.cu`. Le run avait compilé, chargé, annoncé **216 projections
et 0,92 Go sur la carte**, puis réclamé au pilote un symbole que son unité
n'avait jamais contenu.

Corrigé en lisant la **liste de sources**, la seule chose qui décide. Et deux
des trois recherches de noms de `fused_cuda.rs` sont désormais épinglées par un
test portable — ce fichier ne compile pas sur le Mac, donc un nom qu'il tend au
pilote n'était testé qu'en fin de chargement sur carte louée.

Ce test a immédiatement attrapé **une divergence que je n'attendais pas** :
`Planes12x` et `Golay70` *compilent* le noyau segmenté sans pouvoir le
*lancer*, délibérément, pour que le rapport de registres reste un détecteur de
dérive sur ces constructions. **Compilé est un sur-ensemble de lançable**, et
seule l'implication tient. Sur `Tetra48` la disposition, la liste et
`seg_kernel_name` donnaient trois réponses différentes.

**Essai 3** (`6aa23b7721047bf1b03718e8`, 65 s, 0,03 $) :

  `Error: cannot find tensor model.layers.0.self_attn.v_proj.weight`

**C'est la dette du lanceur, et elle hurle.** `FusedModel.int4` est peuplé au
chargement et consommé par personne : `fused_cuda.rs` ne mentionne pas int4.
Les 36 `v_proj` n'ont donc pas de poids, `Proj::pick` retombe sur le tenseur
dense, et le fichier scellé ne le porte pas. Prédit en lisant `pick` sur le
Mac, avant le lancement — la question n'était pas *si* mais *si ça hurle*.

Chiffres *mesurés* au passage, les premiers de VRAM pour ce format :
projections **0,92 Go**, **2,030 b/poids** en comptabilité d'inférence (queue
en binary16), embedding f16 777,9 Mo, **total 1,70 Go**. À comparer aux
1,390 Go *calculés* de `ETAT` — l'écart est l'embedding, ici en f16 quand la
configuration servie le veut en q8.

## É6 — Ce que je mesure à la place, et pourquoi

Le lanceur int4 est la dernière pièce de l'objet servi, et c'est du code
substantiel dans le seul fichier que cette machine ne peut pas compiler. Je ne
l'écris pas en autonomie à cette heure.

**Le fichier Tetra PUR répond aux deux questions de F1e sans une ligne de code
nouvelle** : 252 enregistrements réseau, zéro int4, déjà dans le seau. Il donne
les tok/s et la VRAM du **format**. Ce qu'il ne donne pas, c'est le MMLU de
l'objet servi — mais c'est le recensement à 8 $, retenu de toute façon.

Job `6aa23c235527934177ebf687` : trois arms dans un processus chacun — Tetra
avec embedding f16, Tetra en q8 (la configuration servie), Planes14 en q8 (la
comparaison), 256 jetons chacun. ~0,45 $ estimé, plafond 45 min = 1,35 $.

**Aucune porte de F1e ne se lit là non plus.** La barre de 100,6 tok/s est
celle du protocole de F1e ; ceci en donne un ordre de grandeur sur trois arms
comparables entre eux, dans le même job, sur la même carte.
