# Écarts au pré-enregistrement M0

> Le pré-enregistrement
> [`preregistration-m0-echelles-2026-09-07.md`](preregistration-m0-echelles-2026-09-07.md)
> (sha256 `a75d847d…`) est tamponné. Il ne s'édite pas.

## É1 — La porte du §5 est posée sur une statistique d'échantillonnage, donc elle ne trance rien

**Ce que le préreg pose** : trois seuils sur `σ(ρ*)`, exprimés en fractions de
`(1 − ρ*)`. La première ligne demande `σ ≤ 0,15·(1 − ρ*)`, la deuxième
`σ > 0,40·(1 − ρ*)`.

**Ce qui est mesuré** : `σ/(1 − ρ*)` vaut **0,0816 à 405 blocs par ligne** et
**0,1664 à 106**. La première ligne s'applique à une taille, la troisième — la
zone grise — à l'autre. Le préreg partitionne l'espace des résultats et donne
pourtant deux réponses au même banc.

**Pourquoi, et c'est le défaut** : le balayage de longueur de ligne du journal
montre `σ·√n` constant à 9,9 % près sur un facteur 266 de longueur.
**`σ` suit une loi en 1/√n : c'est une statistique d'échantillonnage.** Sous
cette loi, `σ` croise 0,15·(1 − ρ*) à 120 blocs par ligne et 0,40·(1 − ρ*) à 17.
Les seuils du §5 sont donc des seuils sur la **longueur de ligne**, et sur rien
d'autre.

Les deux revues adverses l'ont trouvé indépendamment et l'ont classé bloquant.

**Ce qui reste établi malgré la porte** : `ρ*` est identique à la sixième
décimale de 6 à 1 600 blocs par ligne. Il n'y a pas de structure par ligne à
exploiter, et M2 — la forme close par ligne — n'a rien à optimiser.

**Ce qu'une porte bien posée aurait demandé** : que `σ` mesurée dépasse la
borne d'échantillonnage `σ·√n` attendue à structure nulle, c'est-à-dire un test
contre des lignes permutées. Le banc le fait en témoin (σ = 0,003198 sur blocs
mélangés contre 0,003204 sur l'ordre naturel) et la réponse est nette : **aucune
structure**. Mais ce témoin n'était pas dans le préreg, donc il ne porte pas de
porte non plus.

## É2 — Le motif de la prédiction signée était faux d'un facteur 8,9

Le §6 dérive `σ` de `sd(1 − cos θ)/√n`, soit 0,000377 à 405 blocs. Mesuré :
0,003358. La décomposition du journal dit pourquoi : `ρ*` par ligne se factorise
en `G · C`, et **`G`, le résidu du code de gain à 1 bit, porte 101,6 % de la
variance ; `C`, le canal angulaire, 1,5 %**. Le motif attribuait `σ` au canal
angulaire.

C'est cette erreur seule qui fait échouer la troisième borne à 106 blocs. Les
deux premières bornes de la prédiction — `ρ*` et `R` — sont justes.

## É3 — Deux des quatre nombres du bras papier sont des identités, pas des mesures

Le §3 demande « les mêmes trois nombres » pour la variante du papier. Deux le
sont par construction : un centroïde de Lloyd–Max est la moyenne de sa cellule,
donc `Σ c(τ − c) = 0` cellule par cellule, donc `Σcτ = Σc²` et `ρ* = 1`, `R = 0`.
Le banc l'imprime et donne le résidu de convergence, −3,7e−15.

Ce qui reste informatif dans ce bras est sa **rétention** et l'écart-type de son
`ρ*` par ligne. Le préreg aurait dû le demander ainsi.

## É4 — Deux lectures du préreg, prises dans le code

**« centroïdes réajustés sur τ/s »** (§3, point 4) : le §3 définit déjà
`τ_p = ⟨x_p, û_p⟩/s`. Lu comme une redondance d'écriture, et les centroïdes sont
ajustés sur `τ_p` tel que défini — la population homologue de `a_p = ‖x‖/s`.
C'est la seule lecture dimensionnellement cohérente, mais c'est une
interprétation.

**`ρ*` global** : la forme du §3, `Σcτ/Σc²`, minimise `J` quand toutes les
lignes partagent une échelle ; le vrai argmin est pondéré par `s²`. Le banc
publie la forme du §3 et imprime l'argmin exact à côté ; l'écart est de 2,3e−6
relatif à 106 blocs et 2,9e−7 à 405. Sur des poids réels, où les échelles de
ligne varient plus que les 1,4 % de dispersion gaussienne, cet écart grandirait.

## É5 — Ce que le banc a ajouté hors du préreg, et pourquoi c'est le chiffre de tête

Le préreg demande quatre nombres. Le banc en publie un cinquième, et c'est celui
qui répond au §2 :

```
  rétention servie, t = 1 ............. 88,77 %
  rétention servie, t = ρ* ............ 89,42 %   +0,6522 pp
  rétention, règle du papier par bloc . 89,44 %   +0,6680 pp
  la constante rend 97,6 % du gain de la règle par bloc
```

Il est hors préreg, donc il **ne porte aucune porte** et ne décide rien. Il est
publié parce que sans lui les quatre nombres demandés ne répondent pas à la
question que le §2 pose.
