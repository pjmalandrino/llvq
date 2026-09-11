# Écarts et scores — préreg F1e recensement du 2026-09-11

Le préreg est tamponné à `de89530` et n'est pas modifié. Ce document se lit
à côté, et il est écrit en deux temps : la fumée maintenant, le recensement
après.

## Premier temps — la fumée (2026-09-11, trois jobs, 0,17 $)

| # | prédit | mesuré | verdict |
|---|---|---|---|
| P1 | la porte servie s'ouvre au premier essai, 70 % | `test -f` vert dans l'image, `mmlu` scoré 57 questions par le noyau, dump avec `# arithmetic=served kernel` (job `6aa40f3a`) | **juste** |
| P2 | le garde-fou à 203 passe, 80 % | argmax 9625 = 9625 | **juste** |

## La borne du garde-fou, écrite de la mesure comme le §4 le prévoyait

Le préreg disait que `max |Δlogit|` serait borné à partir de la fumée, pas
avant. La fumée a rendu **1,8 %** de l'échelle à 203 jetons, contre 0,28 % à
200 et 0,22 % à 800 — et 203 est un paquet de queue de trois rangées, jamais
exécuté sur carte avant ce jour. Deux jobs de plus (0,07 $) ont séparé la
position du mécanisme :

| jetons | queue | max \|Δlogit\| / échelle | argmax |
|---|---|---|---|
| 199 | 3 rangées | 0,31 % | = |
| 200 | aucune | 0,28 % | = |
| 201 | 1 (chemin à une rangée) | 0,14 % | = |
| 202 | 2 rangées | 0,19 % | = |
| 203 | 3 rangées | **1,80 %** | = |
| 204 | aucune | 0,14 % | = |
| 207 | 3 rangées | 0,15 % | = |
| 208 | **aucune** | **2,15 %** (échelle 9,4, paysage plat) | = |
| 223 | 3 rangées, même jeton final que 203 | 1,10 % | = |
| 800 | aucune | 0,22 % | = |

Deux longueurs à trois rangées de queue (199, 207) sont basses, et le plus
haut résiduel de la série est un paquet **plein** (208). **Le résiduel dépend
du jeton final et de son paysage de logits, pas du découpage.** `n_rows = 3`
est propre. Dix garde-fous, dix argmax identiques, enveloppe **0,14 % à
2,15 %** de l'échelle. C'est la borne empirique de ce garde-fou ; ce n'est
pas une borne sur les choix MMLU — quatre logits à un ulp près peuvent
basculer sous 2 % — et c'est précisément ce que P4 et P5 mesurent.

## Écarts au protocole de fumée

Le préreg annonçait un job de fumée ; il y en a eu trois (0,17 $ contre
~0,10 $ annoncés), parce que le premier a rendu un chiffre à expliquer et que
l'expliquer coûtait moins que de le porter dans un recensement à 3,75 $.

## Second temps — le recensement

À écrire après le job : P3 à P7, et le verdict de §5.
