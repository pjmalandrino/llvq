# Rapport d'état — où en est le projet (2026-08-07)

> Photographie complète après la semaine du 04 au 07. Chaque chiffre est
> mesuré, sourcé dans `docs/mesures/`, et donné dans sa comptabilité —
> jamais deux conventions dans une même comparaison. Vulgarisation :
> [`comparatif-simple-2026-08-07.md`](comparatif-simple-2026-08-07.md).

## 1. Le résumé exécutif

En trois jours, **le produit a changé de catégorie sans qu'un seul bit du
modèle publié ne change** : le Qwen3-4B compressé à 2 bits tourne
aujourd'hui à **88,5 tok/s dans 2,60 Go**, contre 43,5 tok/s dans 8,04 Go
pour le moteur de référence — ×2,02 et ÷3,09, à qualité identique au bit
près, chaque étape prouvée avant d'être franchie. En empreinte mémoire
totale, on est passé **sous l'AWQ 4 bits réel** (5,15 b/param contre 5,30),
et le point de fonctionnement suivant (overlay, validé au banc) descendrait
à 4,69. Le front qualité, lui, n'a pas bougé — c'est le pari d'échelle — et
son suspect n°1 a été réfuté proprement cette nuit.

## 2. Les chiffres d'aujourd'hui (L40S, même harnais partout)

| | référence f16 | AWQ 4 bits | **LLVQ aujourd'hui** |
|---|---|---|---|
| disque | 8,04 Go | 2,67 Go | **1,77 Go** |
| VRAM totale | 8,04 Go | 5,30 b/param¹ | **2,60 Go (5,15 b/param)** |
| débit | 43,5 tok/s | jamais mesuré¹ | **88,5 tok/s** |
| ppl wikitext | 12,24 | 13,52 (×1,105) | 16,94 (×1,384) |
| MMLU | 70,3 % | 70,0 % | 55,6 % |

¹ Dans son propre moteur ; l'AWQ n'a jamais tourné dans le nôtre.
⚠️ Toute comparaison VRAM se dit en b/param modèle entier
([`errata-rapport-lot-a-2026-08-06.md`](errata-rapport-lot-a-2026-08-06.md)).

**Le ×2,02 de débit est attribué phase par phase**
([`mesures/phases-2026-08-07.txt`](../mesures/phases-2026-08-07.txt)) : ~26 ms
par token viennent d'un défaut de **notre bras dense** — `Head::project`
appelle `broadcast_matmul`, qui recopie les 778 Mo du vocabulaire à chaque
token (le `TODO` est dans le code de candle, la copie est chronométrée) —
que notre noyau q8 ramène à 0,6 ms ; ~2,9 ms viennent du noyau Leech sur les
projections. 🚨 **Le défaut est dans la primitive, pas dans les modèles de
candle**, qui atteignent leur tête par `Linear` et évitent ce chemin
([candle#3871](https://github.com/huggingface/candle/issues/3871)) : la
baseline handicapée est la nôtre, donc le ×2,02 ne porte pas seul.
Formulation double au comparatif : ×2,02 contre notre bras dense tel quel,
~×1,4 contre ce même bras corrigé (estimé des phases), et ×1,12 à tête
identique pour le noyau seul.

## 3. L'échelle des formats — trois points de fonctionnement, tous vérifiés

| layout | b/poids payload | vs FP16 (banc) | statut |
|---|---|---|---|
| Slot32 (ancien) | 5,510 | 1,89× | remplacé, conservé en repli |
| **Planes14** | 4,804 | **2,16×** | **en production** (défaut) |
| **Planes12x** (overlay) | **4,342** | 2,01× | validé au banc, qualité exacte |
| Golay70 (E2) | 3,589 | 1,31× | **mesuré, écarté comme point produit** — sous le critère de 1,6× posé d'avance : le décodage à double coset borne le noyau en calcul (195 Go/s effectifs). Le résultat de *format* tient (l'information Golay est recomputable à qualité exacte, prouvé sur les 150,7 M blocs) ; le point de fonctionnement, non ([`mesures/e2-golay70-bench-2026-08-07.txt`](../mesures/e2-golay70-bench-2026-08-07.txt)) |

La découverte fondatrice de la semaine : le one-hot de Slot32 ne payait
rien, il coûtait — le recodage binaire a rendu le format **plus petit ET
plus rapide** (Go/s constants, le temps tombe comme les octets). Le plafond
L≤4 sec est mort (+4,75 % de ppl mesuré au swap) ; l'overlay le remplace à
qualité exacte. Bijections et sweeps intégraux (150,7 M blocs) à chaque
étape ; zéro spill sur les quatre noyaux.

## 4. Le front qualité — un suspect réfuté, le pari d'échelle intact

Le −14,7 pp de MMLU face au 4 bits reste LE point faible du 4B. Cette
semaine l'a instruit sans le résoudre, et c'est un progrès :

- **Rotation de sortie** : morte (Table 9 : effet ≈ 0 à Input fixé).
- **Calibration** (volume/corpus) : plafonnée par l'oracle (−1,6 % au
  maximum théorique) — le run à 25 $ annulé.
- **Design C** (rétraction libre + résolution close) : **réfuté à pleine
  profondeur** (0.6B, 28 blocs : ×1,99 de ppl, gate automatique, 0 $ GPU).
  Deuxième occurrence du motif « proxy local meilleur, composition
  désastreuse » après group_scales : **la rigidité de norme de la
  rétraction sphérique est porteuse à profondeur** — un fait de méthode.
- Restent : la config de gain (1,4 pp), la composition du corpus (P18),
  la compensation post-hoc (EoRA +4-11 pp publiés), le FT échelles
  (+2,1 pp), et **l'axe d'échelle** — le 8B se dégrade déjà moins (×1,267
  contre ×1,386), jamais mesuré en MMLU. À 2,60 Go pour un 4B, un 70B
  tiendrait dans ~24-30 Go selon le layout : c'est là que le 2 bits
  redevient nécessaire.

## 5. La méthode, et ce qu'elle a coûté

Chaque marche : spécification figée → implémentation par agents →
**revue adversariale systématique** (mutants rejoués, sweeps relancés,
sondes indépendantes) → mesure sur carte avec gates automatiques. Les
revues ont attrapé avant la carte : un fichier tronqué plausible, un signe
d'échelle perdu à la re-projection, deux nits de comptabilité, une erreur
grave dans le rapport du lot A (deux quatre-bits confondus). Les gates ont
bloqué un run 4B de 4 h sur un design réfuté.

**Coût GPU de la séquence C1 → phases : 1,64 $** (C1 0,08 · A/B
branchement 0,33 · nuit 0,90 · phases 0,33 ; un essai d'invocation à
0,00 $). Le lot A en avait coûté 2,19. Tout est commité, poussé, et consigné dans
`docs/mesures/`.

## 6. Ce qui reste, par ordre

1. ~~E2 Golay~~ **tranché le jour même** : 3,589 b/poids réels et
   reconstruction exacte, mais **1,31× vs FP16** — sous le critère (1,6×).
   Le coût ALU du double coset l'emporte ; l'échelle s'arrête proprement à
   Planes14/Planes12x. Pistes si quelqu'un veut rouvrir un jour :
   spécialiser les warps par coset, ou payer le XOR seulement côté pair —
   notées, non poursuivies.
2. **Choix produit** : Planes14 (vitesse) ou Planes12x (bits) comme défaut
   — décision à prendre sur les besoins réels, les deux sont prêts.
3. **Qualité** : reprendre par les suspects survivants — à décider à la fin,
   comme convenu.
4. **Publication** : le README et CLAUDE.md portent déjà les chiffres ; une
   passe de mise en cohérence des docs anciens (fiche-4b, face-au-4-bits)
   reste à faire avant toute communication externe.
