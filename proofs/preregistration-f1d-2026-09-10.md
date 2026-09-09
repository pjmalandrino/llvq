# Pré-enregistrement — F1d : le décodage Tetra servi, bras du banc publié, deux cartes

**Écrit, commité et TAMPONNÉ le 2026-09-10, AVANT le lancement.** Go permanent
de l'opérateur du 2026-09-09 au soir : « Tu peux enchaîner sur les 6.8 et 6.9
avec 20 $ de plafond directement quand terminé. »

Ce fichier ne s'édite plus. Ce qui le corrige va dans un `-ECARTS.md`.

**Coût : ~0,60 $** (*estimé*, deux jobs — `rtxpro6000x1` puis `l40sx1` — de
~12 min de calcul chacun : trois minutes de transcodage de 252 matrices, puis
sept tours sur douze bras. Repère : le `planesbench` du 2026-08-06 a coûté
0,08 $ pour 3 min à six bras). Plafond de timeout **40 min par job**, soit
**~2,00 $ au pire**. Aucun encodage, aucun téléversement : les deux fichiers
sont déjà dans le seau. Total projet avant : **134,26 $**. Plafond wave 3 :
**20,00 $**, dépensé **0,00 $**.

---

## §1 — La question

`t(tetra48) / t(planes14)`, formé **tour par tour dans un seul processus**, sur
un vrai fichier, avec les vraies étiquettes, contre QTIP et AWQ dans leurs
propres grilles et FP16 comme témoin.

La tuerie du 2026-09-08 a mis les deux décodages dans un processus, mais sur un
**flux synthétique** à étiquettes uniformes. Ce qu'elle ne dit pas, et que ce
banc dit : le mélange de classes d'un artefact réel, les sept formes du modèle,
252 lancements par tour, et une unité de traduction qui porte douze bras au
lieu de huit.

## §2 — Ce qui est mesuré, et sur quoi

**Deux fichiers, un processus.** Tetra et Planes14 ne partagent pas de
fichier — alphabets de codes différents — donc `planesbench` prend deux chemins
et lit chaque format depuis le sien. Le rapport reste intra-processus, tour par
tour : la règle dure 5 porte sur les cartes et les piles, pas sur les fichiers.

- bras balle : `tetra-4b-2026-09-06/qwen3-4b-planes14.bin` (1 770 527 533 o)
- bras Tetra : `tetra-4b-2026-09-06/qwen3-4b-tetra.bin` (1 770 529 149 o)

**Les deux sont déjà dans le seau `Pier-Jean/jobs-artifacts`** (vérifié par
`hf buckets ls` le 2026-09-10, règle dure 9). Rien à encoder, rien à téléverser.

### Le choix du fichier Tetra, et pourquoi ce n'est pas le fichier servi

L'objet servi arbitré le 2026-09-08 est **Tetra + `v_proj` en int4** — le
fichier `qwen3-4b-tetra-q5.bin` de l'étape 6.0, qui n'existe que sur le Mac.
F1d ne l'utilise pas, **délibérément**, et c'est une décision prise seul faute
de pouvoir demander :

1. **La porte de F1d est une question de noyau**, pas de produit : « le
   décodage servi ajouté comme bras au banc de comparaison publié »
   (ROADMAP §2.2 quinquies, ligne 6.8). Le temps d'un noyau ne dépend pas de la
   qualité des poids qu'il lit.
2. **Le support serait inégal.** Le fichier mixte porte 216 enregistrements
   Tetra et 36 int4 ; un tour du bras Tetra y compterait 216 lancements contre
   252 pour Planes14. Le fichier Tetra pur en porte 252 : **support égal, et le
   rapport est comparable lancement pour lancement.**
3. **Le produit servi est la question de F1e**, qui tourne `fusedrun` sur le
   fichier scellé et mesure les tok/s et le MMLU. C'est là que le fichier mixte
   a sa place, et c'est là qu'il faudra le téléverser.

Conséquence assumée : F1d ne dit rien sur `tv_q4_h`, qui n'a toujours jamais
tourné sur carte.

## §3 — Les portes, et laquelle tue

| porte | seuil | lecture |
|---|---|---|
| **débit** | `t(tetra48) ≤ t(planes14)` en médiane, dans le processus | **kill si plus lent pour moins d'octets** |
| **mémoire** | `≤ 2,20 b/poids` noyau, dénominateur du bras | informe |
| registres | `num_regs ≤ 64`, `local_bytes = 0` sur `tv_f1r_v3g` | arrête le run (contrat) |
| exactitude | `worst_error < TOL = 1e-5` contre sa propre référence f64 | arrête le run |

**Un kill est l'affaire de l'opérateur** (règle du 2026-09-05). Ce banc mesure
et écrit ; il ne tranche pas.

## §4 — La tuile : trois colonnes, décidées d'avance

`TILE_BLOCKS` est une constante injectée par l'hôte : zéro bit, sortie
bit-identique. Le balayage du 2026-09-09 a mesuré que l'optimum **dépend de
l'architecture** — 64 sur sm_89, 32 sur sm_120, contre 128 servi — et que c'est
ce qui explique tout l'écart entre les deux cartes.

Mesurer F1d à 128 seul rendrait un rouge qu'on sait être un artefact de
réglage. Le mesurer à 32/64 seul mesurerait une configuration qui n'est pas
servie. Donc **trois lancements par carte**, dans le même job :

| colonne | `LLVQ_TILE_BLOCKS` | ce qu'elle répond |
|---|---|---|
| **servie** | non posé (128) | comparable à tous les chiffres publiés |
| **optimum** | `auto` (64 sur L40S, 32 sur RTX) | quel format sert le plus vite |
| **contrôle** | l'autre valeur mesurée | Planes14 bouge-t-il, et de combien |

**La porte du §3 se lit sur la colonne SERVIE.** Les deux autres informent une
décision d'opérateur — basculer le défaut par carte — qu'aucune mesure ne prend
à sa place.

Le balayage coûte 34 s pour trois points (*mesuré*, 2026-09-09). Trois
lancements de `planesbench` coûtent trois transcodages, soit ~9 min de plus par
carte ; c'est compté dans l'estimation du §0.

## §5 — Les prédictions signées

Écrites avant, et fausses si elles sont fausses.

| # | prédiction | intervalle |
|---|---|---|
| P1 | `tetra48 / planes14` brut, RTX, tuile 128 | **1,35×** [1,10 ; 1,65] |
| P2 | `tetra48 / planes14` brut, L40S, tuile 128 | **0,70×** [0,55 ; 0,90] |
| P3 | RTX à tuile 32, le rapport tombe sous | **0,95×** |
| P4 | Planes14 bouge d'une tuile à l'autre de moins de | **±4 %** |
| P5 | b/poids noyau du bras Tetra | **2,158** [2,10 ; 2,20] |
| P6 | `tv_f1r_v3g` : registres et débordement | **40 regs, 0 local** |

P1 et P2 sont plus proches de 1 que les R du banc synthétique (1,4643 et
0,5784) parce que ceux-là sont des rapports **à tête égale**, plancher retranché,
et que ceux-ci sont bruts : le plancher de lancement est commun aux deux bras
et pousse tout rapport vers 1. La ligne à tête égale du banc est imprimée à
côté, règle dure 4.

## §6 — Ce que ce banc ne mesure pas

- **Aucun tok/s.** 48 % d'un jeton est hors des matmuls (attribution du
  2026-08-05), donc un rapport de noyau se comprime de bout en bout. C'est F1e.
- **Aucune qualité.** Le fichier Tetra pur est celui à 53,49 MMLU ; le produit
  servi est à 56,95. F1d ne lit ni l'un ni l'autre.
- **`tv_q4_h` n'a toujours pas tourné sur carte.**
- **La section A3 est hors du run** : elle dimensionne ses lancements sur la
  constante et `occ_pers` change de branche au seuil de la tuile, donc elle est
  refusée par nom hors tuile servie.

## §7 — Les décisions prises seul, faute de pouvoir demander

L'opérateur dort. Ces quatre-là sont miennes et se contestent au réveil :

1. **Le fichier Tetra pur plutôt que le fichier mixte** (§2).
2. **Les trois colonnes de tuile**, la porte se lisant sur la servie (§4).
3. **Le dénominateur b/poids est celui du bras**, pas celui du modèle — un bras
   qui ne lit pas toutes les matrices divisait par des poids qu'il ne lit pas.
   Sur le fichier Tetra pur les deux dénominateurs coïncident, mais le code
   change et c'est écrit ici.
4. **RTX d'abord, L40S ensuite** : la file de la RTX était sous la minute le
   2026-09-08, celle de la L40S à 3 h 39 le 2026-09-09.

## §8 — Ce qui a changé dans le code avant ce banc, et qui pourrait le fausser

Écrit ici pour que la relecture puisse s'y attaquer.

- `planesbench` lit un second fichier par `read_record` et apparie **par nom**.
- Le bras 17 est dispatché ; il ne l'était pas, et un `planesbench` nu paniquait.
- Le dénominateur b/poids est devenu celui du bras.
- La tuile est résolue avant la source et vérifiée dans le texte assemblé.
- L'unité composite de `cuhcheck` est assemblée par le texte et non par
  `#include` : cinq mutants qui survivaient sont tués.
- Aucun de ces changements ne touche un bras publié : les fragments NVRTC sont
  **ajoutés en fin d'unité**, et l'ordre relatif des fragments existants ne
  bouge pas d'un cran. La sha256 de l'unité bouge pour TOUS les bras, donc
  aucun chiffre d'un run antérieur ne se reporte : ce job re-mesure tout.
