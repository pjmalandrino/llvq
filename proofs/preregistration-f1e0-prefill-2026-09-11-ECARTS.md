# Écarts et scores — préreg F1e §0 quater du 2026-09-11

Le préreg est tamponné et n'est pas modifié. Ce document est ce qui se lit à
côté.

## Les cinq prédictions, notées

| # | prédit | mesuré | verdict |
|---|---|---|---|
| P1 | garde-fou passe, 75 % | argmax 785 = 785 aux deux longueurs | **juste** |
| P2 | 3,6 ms/jeton à 800 [2,5 ; 4,6] | **3,900** | **juste**, dans l'intervalle |
| P3 | 256 jetons identiques au dense, 85 % | identiques | **juste** |
| P4 | rien ne refuse au premier essai, 60 % | rien n'a refusé | **juste** |
| P5 | décodage inchangé, 100,8 ± 2 tok/s | **101,5** [101,0–101,9] | **juste** |

Cinq sur cinq. C'est le meilleur score de ce chantier et il mérite d'être
tempéré : P1, P3 et P4 étaient des prédictions de « ça marche », écrites après
trois mutants tués et un conteneur qui type-checke. La seule qui portait un
risque réel était P2, et son point était à 3,6 pour un mesuré à 3,900 — juste
par l'intervalle, pas par le point.

## Ce qu'il faut noter à côté de P5

101,5 tok/s contre 100,8 : **les intervalles ne se recouvrent pas**
([101,0 ; 101,9] contre [100,4 ; 100,8]). P5 est juste telle qu'écrite, à
± 2 tok/s, et la non-superposition n'autorise aucune conclusion : deux jobs,
deux processus, deux instances de carte. La règle 5 interdit d'en former un
rapport. Le chemin à une rangée est inchangé dans le code — `prepare_rows`
court-circuite vers `prepare`, `forward_rows` vers `forward_with` — et l'écart
est du bruit inter-job.

## Le raisonnement de P2, vérifié plutôt que le seul chiffre

Le préreg pariait 6,5 ms d'économie par paquet en supposant ~5 µs par
lancement. Mesuré : **6,22 ms**, soit 4,80 µs par opération supprimée. Ce
n'est pas seulement P2 qui tombe juste, c'est le mécanisme invoqué pour la
prédire. C'est ce qui rend la projection du poste suivant (`regroup`,
4,84 ms par paquet) crédible plutôt qu'inventée.

## Aucun écart de protocole

Mêmes deux longueurs, mêmes cinq passes après une passe jetée, même fichier,
même carte, mêmes drapeaux. Coût 0,12 $ contre ~0,15 $ annoncés.

## L'écart déclaré par le préreg lui-même

Son §0 constate que la mesure du 2026-09-10 (5,507 ms/jeton, job 6aa32150) a
été lancée hors préreg. Il reste constaté ; rien ne le répare.
