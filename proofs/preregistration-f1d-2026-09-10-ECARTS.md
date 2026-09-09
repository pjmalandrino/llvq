# Écarts au pré-enregistrement F1d du 2026-09-10

Le préreg est tamponné et ne s'édite pas. Ce qui le corrige est ici, daté.

---

## É1 — Le coût du §0 est sous-estimé d'un facteur 3

**Annoncé : ~0,60 $ pour les deux cartes. Révisé : ~2,00 $.**

L'erreur est une erreur de comptage, pas de prix. Le §4 demande **trois
colonnes de tuile par carte**, et la tuile est un `#define` : trois colonnes
sont **trois processus**, donc trois transcodages complets des 252 matrices, pas
un transcodage suivi de trois séries de tours.

| | estimé §0 | révisé |
|---|---|---|
| par processus | — | ~8 min (6 de construction, 2 de tours) |
| par carte | ~12 min | ~26 min (3 processus + démarrage) |
| RTX PRO 6000 à 2,75 $/h | — | ~1,19 $ |
| L40S à 1,80 $/h | — | ~0,78 $ |
| **total** | **0,60 $** | **~2,00 $** |

Plafond de timeout porté à **60 min par job**, soit **4,55 $ au pire** pour les
deux. Toujours très en deçà des 20 $ de la vague 3.

Les trois colonnes sont maintenues : réduire à deux économiserait 0,65 $ et
perdrait la colonne qui dit si la **ligne de base** doit bouger — la seule
question de tuile que le balayage du 2026-09-09 n'a mesurée que sur un flux
synthétique.

## É2 — Un défaut trouvé APRÈS le tampon, et corrigé avant la première mesure

Le §8 énumère ce qui avait changé dans le code avant le banc. Il en manquait
un, trouvé une heure après le tampon en faisant tourner `rtbits` sur le fichier
mixte pour une raison sans rapport.

**Le bras Tetra facturait son flux seul.** Tous les autres bras du banc
facturent flux + queue + échelles de ligne (`SlotArm`, `PlanesArm`, `P12Arm`
ajoutent tous `(d_out · tail_w) · 4 + d_out · 4`). Sur le 4B ces données de
côté pèsent 69,75 Mo contre 908 Mo de flux : la colonne b/poids de Tetra aurait
imprimé **~2,00 au lieu de ~2,154**, dans la colonne même que lit la porte de
2,20.

Corrigé au commit `205b9d5`, **avant tout lancement**. La prédiction P5
(2,158 [2,10 ; 2,20]) était écrite contre la comptabilité correcte et reste
donc telle quelle ; c'est le code qui ne la respectait pas.

Le même passage a confirmé, par une seconde route, la part de `v_proj` :
94 371 840 de 3 633 315 840 poids = **2,5974 %**.

## É3 — L'image, et ce qu'elle porte

L'image du Space est reconstruite au commit **`205b9d5`**, qui inclut É2. Les
mesures de F1d sont attribuées à ce commit et à aucun autre.

La construction précédente, au commit `88c2294`, a été abandonnée sans avoir
servi : elle portait le défaut de É2.

## É4 — La colonne b/poids du banc n'est pas la comptabilité de la porte

Constaté en écrivant É2, et il faut le dire avant de lire un tableau.

`rtbits` sur le fichier mixte imprime **2,2030 b/poids noyau** (queue en f32),
au-dessus de la porte de 2,20 — parce qu'il compte les 36 matrices `v_proj`
à 4,25 b/poids. Le banc, lui, mesure le **fichier Tetra pur** et sa colonne
porte les octets qu'un bras lit pour une matrice.

Ce sont deux comptabilités et elles ne se comparent pas. La porte de 2,20 se
lit sur celle du banc, pour le fichier que le banc a lu, et le dossier ne doit
pas mélanger les deux — c'est la règle des trois comptabilités jamais mêlées.
