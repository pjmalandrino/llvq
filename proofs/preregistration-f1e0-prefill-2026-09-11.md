# Pré-enregistrement — F1e §0 quater : ce que vaut la rotation par lots

Écrit et tamponné **avant** le premier lancement de la mesure, au commit
`0c2d798`. Règle dure 2.

## §0 — Un écart à déclarer d'abord

La mesure du 2026-09-10 au soir (job `6aa321505527934177ec2d0d`, 5,507 ms par
jeton de prompt) a été lancée **hors préreg**. Le préreg de F1e §0 couvre « le
chemin servi tourne-t-il sur une carte », pas le coût d'un préremplissage.
C'est un manquement à la règle 2, constaté ici plutôt que passé sous silence.
Ce document couvre la mesure suivante, celle qui a un avant et un après.

## §1 — Ce qui est mesuré

Le même objet servi, la même carte, les mêmes deux longueurs de prompt qu'au
job `6aa32150` : 200 et 800 jetons, cinq passes après une passe jetée. Ce qui
a changé entre les deux est **le code**, au commit `0c2d798` :

* la rotation prend les rangées dans sa grille (`rot_apply_rows`), une passe
  au lieu de quatre par groupe et par paquet ;
* le `Tensor::cat` qui empilait les rangées avant le matvec disparaît, parce
  que la rotation produit désormais la forme voulue.

Compte d'opérations par paquet de 4 rangées, *calculé* : **1 800 → 504.**

## §2 — Ce qui compte comme un succès

Le garde-fou d'abord, le chiffre ensuite. Si les deux bras du garde-fou
choisissent des jetons différents, il n'y a pas de mesure : il y a un défaut.

## §3 — Les prédictions signées

| # | quantité | prédiction |
|---|---|---|
| P1 | le garde-fou passe (même argmax, les deux bras) | **oui, 75 %** |
| P2 | ms par jeton de prompt à 800 jetons | **3,6** [2,5 ; 4,6] |
| P3 | les 256 jetons restent identiques au bras dense | **oui, 85 %** |
| P4 | quelque chose refuse au premier essai | **non, 60 %** |
| P5 | débit de décodage (256 jetons, tok/s) | **inchangé**, 100,8 ± 2 |

**Le raisonnement de P2, pour qu'il soit jugeable.** À 800 jetons le
préremplissage dure 4 364 ms, soit 21,8 ms par paquet de 4 rangées. Ce que le
changement supprime, ce n'est **pas de l'arithmétique** — les quatre rangées
subissent la même transformée, dans quatre blocs au lieu de quatre
lancements — c'est 432 lancements de rotation et 864 recopies par paquet. À
~5 µs de surcoût par lancement, cela fait 2,2 + 4,3 = **6,5 ms sur 21,8**, donc
15,3 ms par paquet et 3,8 ms par jeton. J'arrondis à 3,6 en supposant que les
recopies coûtent un peu plus qu'un lancement vide.

**P5 est là pour être fausse si je me suis trompé.** Le décodage est une
rangée : `prepare_rows` court-circuite vers `prepare`, `forward_rows` vers
`forward_with`. Si le débit de décodage bouge, c'est que le chemin à une
rangée n'est pas resté identique, et le chiffre publié de 100,8 tok/s tombe
avec lui.

## §4 — Ce que je NE décide pas ici

Le batchage de la **sortie** (`regroup`, ~200 000 recopies par
préremplissage de 800 jetons) est écrit nulle part et n'est pas dans cette
mesure. L'empiler avant de mesurer rendrait les deux illisibles.

## §5 — Coût

Un job l40sx1, plafond 20 min, 0,60 $ au pire, ~0,15 $ attendu. Cumul de la
vague 3 avant ce job : 4,29 $ sur 20 $.
