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

## Écart majeur, déclaré le 2026-09-11 — l'objectif est le SPLIT COMPLET

Le préreg est tamponné et ne bouge pas ; ceci le corrige à côté, et c'est le
plus gros écart du document.

**L'opérateur rappelle que le postulat de base du chantier est MMLU sur les
14 042 questions, pas sur l'échantillon de 2 280.** Le préreg a été écrit sur
2 280 parce que c'est le plan de toutes les barres déjà publiées du dépôt ; il
n'a pas remonté au postulat. C'est une erreur de rédaction du préreg, pas une
décision prise puis changée.

**Ce que le job en cours reste :** `6aa414dd` mesure deux bras sur 2 280. Il a
été laissé finir sur instruction. Sa valeur n'est pas le chiffre publiable —
c'est **P4**, l'écart apparié noyau − dense sur poids identiques, qui dit si
le noyau CUDA calcule juste. Un split complet par un noyau faux coûterait
19 $ pour rien. Le recensement 2 280 est donc le contrôle, pas le résultat.

**Ce que le split complet demande, chiffré sur la pente mesurée**
(3,929 ms/jeton, 9,76 M jetons de prompt contre 1,38 M, ×7,06) :

| bras | durée | coût |
|---|---|---|
| Tetra, noyau servi | 10,65 h | 19,17 $ |
| Tetra, dense, même fichier mixte | 2,67 h | 4,80 $ |
| f16 (rejeu obligatoire, voir ci-dessous) | 2,67 h | 4,80 $ |
| Planes14 (rejeu obligatoire) | 2,67 h | 4,80 $ |
| **total** | **18,7 h** | **33,57 $** |

**Pourquoi f16 et Planes14 doivent être rejoués.** Les dix-neuf dumps de
`docs/data/mmlu-dumps/` portent tous `# limit=40`, 2 280 questions, empreinte
`65dcd53655e8bfa5`. `bin/mmlupair` refuse deux dumps de plans différents
(`mmlupair.rs`, champs `limit` et `fingerprint`). Un Tetra sur 14 042 ne
s'apparie donc avec **aucun** chiffre existant du dossier : ni les 70,32 de
f16, ni les 55,59 de Planes14, ni les 56,95 de Tetra. Le split complet n'est
pas « le même tableau en plus grand », c'est un dossier neuf.

**Deux leviers avant de payer**, tous deux déjà chiffrés dans
`docs/audit-livrable-2026-09-11.md` : `regroup` (a), −20 % *estimé*, code
portable ; et le noyau int4 par lots, −8 % *estimé*. Ensemble ils ramènent le
bras noyau de 19,17 $ à **~13,80 $** et le total à **~28 $**.

**Le plafond de la vague 3 est 20 $, dont 4,58 $ dépensés.** Le programme
complet ne rentre pas. C'est une décision d'opérateur : relever le plafond,
ou réduire le programme (par exemple noyau + dense en full, sans rejouer f16
et Planes14, et le tableau à quatre reste au plan 2 280 à côté).
