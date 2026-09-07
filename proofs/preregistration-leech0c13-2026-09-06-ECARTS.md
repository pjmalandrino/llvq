# Écarts au pré-enregistrement du chantier 3

> Le pré-enregistrement
> [`preregistration-leech0c13-2026-09-06.md`](preregistration-leech0c13-2026-09-06.md)
> (sha256 `9ecd46f8…`) est tamponné. Il ne s'édite pas.

## É1 — La règle porte sur un point, et l'intervalle chevauche deux lignes

Le §5 partitionne sur **M**, l'estimation ponctuelle du MMLU micro. Mesuré :
M = 54,67, donc la troisième ligne, « le codebook est innocenté ».

Ce que la règle ne dit pas, et qu'il faut lire avec : l'IC95 apparié de
`leech0c13` contre Tetra est **[−1,54 ; +3,98]** autour de +1,19 pp, donc M
pourrait valoir jusqu'à 57,5 sous rééchantillonnage. **La frontière entre la
troisième ligne et la deuxième, 55,5, est dans l'intervalle.**

Ce qui reste robuste : la première ligne, 58,5, est **hors** de l'intervalle.
La conclusion « le codebook ne porte pas l'essentiel des 5,1 points » tient. La
conclusion « le codebook ne porte rien du tout » ne se lit pas sur ce banc.

Une règle mieux écrite aurait posé ses seuils sur l'intervalle et non sur le
point. C'est à corriger dans le prochain préreg.

## É2 — L'encodage a pris 3 h 55 au lieu des ~4 h 30 annoncées

386 s/bloc contre les 418 estimés en cours de route. Sans conséquence : aucun
seuil du préreg ne porte sur la durée, et le coût annoncé était 0 $.

## É3 — Le §4 ne s'est pas déclenché, et c'est une information

Le préreg prévoyait de tout republier en b/param si le débit de `leech0c13`
s'écartait de plus de 0,05 b/poids de celui de `leech1c12`. Mesuré : **2,0702
des deux côtés, à la quatrième décimale**, 48 bits par bloc. Le garde-fou
n'était pas superflu — il était impossible à écarter d'avance, et il est levé
par la mesure.
