# G6 — noyau fusé Metal : brief de démarrage

> Document de reprise pour une session dédiée au noyau. Tout ce qui précède
> (G1–G5) est acquis et n'a pas besoin d'être relu : voir `CLAUDE.md` si un
> détail manque.

> 🗓️ **BANDEAU D'ÉTAT — dernière revue le 2026-08-08. G6 est franchi et
> dépassé.** Le noyau existe sur Metal (banc) **et sur CUDA** (branché dans le
> modèle, `bin/fusedrun`), le layout de référence est `Planes14` et non
> `Slot32`, et le run mentionné ci-dessous est terminé depuis le 2026-07-31.
> Ce document reste utile pour son exposé du problème, pas pour son état.
> À jour : [`rapport-etat-2026-08-07.md`](rapport-etat-2026-08-07.md).

> ~~**Un run tourne peut-être encore**~~ *(terminé)* : voir
> `docs/run-de-nuit.md` (`Λ₂₄(12)` + 1 bit de gain sur Qwen3-4B, ~3,5 h). Son
> résultat n'était pas bloquant pour G6 — il a affiné le débit publié, il n'a
> pas changé le noyau. C'est lui qui a produit le fichier scellé publié.

## Où en est le projet

**G1–G5 verts.** Λ₂₄ + Golay, recherche NN exacte m ≤ 13, indexage bijectif
48 bits, source gaussienne à 92 % de rétention Shannon, et Spherical GPTQ
validé sur Qwen3-4B : **14,9104 de perplexité wikitext-2 à 2,1117 bits/poids**,
contre 17,04 pour QTIP et 15,54 pour la meilleure config sans fine-tuning du
papier. Calibration hors domaine, protocole du papier.

**Ce qui manque, et c'est tout ce qui reste :**

1. **L'artefact 2 bits réel.** Le quantifieur produit des reconstructions ; le
   format d'index 48 bits existe et est testé (`llvq-search/src/index.rs`,
   aller-retour exhaustif sur Shell(2) + 2 M d'indices aléatoires), mais rien
   ne l'écrit. Le 1,74 Go est calculé, pas produit.
2. **Le noyau fusé.** Aucun gain à l'inférence aujourd'hui — zéro.

## Pourquoi le noyau, et quel est l'enjeu exact

La génération token par token est **limitée par la bande passante mémoire**,
pas par le calcul : produire un token exige de lire *tous* les poids.

```
Qwen3-4B en FP16 : 8 Go/token ÷ ~400 Go/s = 20 ms → ~50 tokens/s au plafond
Qwen3-4B en 2 bits : 1 Go/token             =  2,5 ms → ~400 tokens/s
```

Mais il faut des nombres pour multiplier, pas des index. Deux voies :

- **naïve** — décoder tous les poids en mémoire puis multiplier : on écrit 8 Go
  et on relit 8 Go, **c'est pire qu'avant** ;
- **fusée** — lire le 1 Go d'index, décoder chaque bloc de 24 **dans les
  registres**, multiplier immédiatement, ne jamais écrire les poids décodés.

Seule la seconde donne le gain. C'est tout le travail de G6.

## Le budget, et il est serré

```
151 M blocs de 24 poids par passe ÷ 2,5 ms  =  60 G blocs/s
GPU M3 Max ≈ 4 Topérations/s
→ ~65 opérations GPU par bloc de 24 poids
```

Or décoder un bloc (annexe A du papier, implémenté dans `index.rs`) demande :
identifier la coquille par recherche dans 12 cumuls, identifier la classe
parmi **383**, puis dérouler un rang de permutation sur 24 positions — du
calcul en base factorielle avec **divisions entières**, ce qu'un GPU déteste.

Tenir en 65 opérations en multi-coquilles est douteux. **C'est le risque
principal du projet, et il n'est pas levé.**

## La décision de conception à trancher en premier

**Coquille unique plutôt qu'union.** Mesuré sur source gaussienne
(`cargo run --release -p llvq-bench --bin llvq-bench`) :

| code | bits/dim | rétention | classes | norme |
|---|---|---|---|---|
| union `norm(Λ₂₄(12))` + 1 bit de gain (meilleur du papier) | 2,0000 | 92,14 % | 383 | variable |
| **coquille 12 seule + 1 bit de gain** | **1,9584** | **92,24 %** | **79** | **constante** |
| coquille 13 seule + 1 bit de gain | 2,0113 | 92,33 % | 82 | constante |

> ⚠️ Rétentions révisées le 2026-08-01 (§A5) : 92,81 → 92,24 et 92,83 → 92,33.
> Le banc codait le gain sur la projection, la production le code sur la norme
> du bloc. **La marge sur le papier tombe de 0,67 à 0,10 point** : la coquille
> unique fait maintenant *jeu égal* en rétention, pas mieux. L'argument tient
> toujours, mais il repose désormais sur le débit et le matériel, plus sur la
> qualité.

Trois conséquences pour le noyau : plus de recherche de coquille, **4,8× moins
de classes**, et une **norme constante** — donc un facteur d'échelle fixe entre
produits scalaires, donc plus de rééchelonnage des accumulations
intermédiaires. C'est probablement la différence entre faisable et pas
faisable.

⚠️ **Non vérifié sur de vrais poids après GPTQ.** Le papier mesurait une
distance angulaire sur source radialement uniforme, pas une rétention MSE ;
et les poids ne sont pas gaussiens. **À valider avant d'écrire une ligne de
noyau** — c'est un run de 3,5 h avec un quantifieur restreint à une coquille.

## Le premier pas : mesurer, ne pas construire

**Ne pas écrire le noyau complet d'emblée.** Écrire un micro-banc Metal qui ne
fait *que* décoder N index en vecteurs, et le chronométrer.

- s'il tient ~60 G blocs/s, le noyau complet vaut l'investissement ;
- s'il plafonne dix fois en dessous, on le sait en **une journée** et on
  ajuste — coquille unique, ou format d'index simplifié qui échange un peu de
  débit contre un décodage trivial.

Même principe que `cholbench`, qui a montré que le Cholesky maison plafonnait
à 1 G mult-add/s et justifié `faer` (×105) avant qu'on y passe des jours.

## Matériel : Metal

Décidé. La machine de dev est un M3 Max, et c'est le bon choix de fond : la
**mémoire unifiée n'a pas de plafond VRAM**. Un 70B à 2 bits fait ~20 Go, donc
il tient ; sur NVIDIA grand public il faudrait deux cartes. C'est l'argument
souveraineté.

## Ce sur quoi s'appuyer

| Élément | Où | État |
|---|---|---|
| `Indexer::encode/decode` (48 bits) | `llvq-search/src/index.rs` | testé, aller-retour exhaustif |
| Contrat de stabilité du format v1 | en-tête de `index.rs` | documenté |
| Procédure de déquantification | `docs/llvq-paper-notes.md`, annexe A | transcrite |
| Repères de vitesse du papier (Table 7) | idem | FP16 16,3 µs · leur noyau 11,94 µs |
| Modèle quantifié sauvegardé | `artifact::save/load` | fonctionne |

⚠️ **Le format v1 est épinglé sur la boule m ≤ 13.** Passer à une coquille
unique casse la compatibilité des fichiers déjà quantifiés — décision à
assumer explicitement, ce n'est pas un détail d'implémentation.

## Barre de réussite, et la nuance qui compte

Le trophée serait de battre le FP16. **Mais ce n'est pas la barre du utile.**
Un 70B en FP16 fait 140 Go : il ne tourne nulle part en local, à aucune
vitesse. À 2 bits il fait 20 Go et il tient.

Donc un noyau à **0,8× la vitesse FP16 est déjà un gain net**, parce qu'il rend
possible ce qui était impossible. La vitesse ne devient une question qu'une
fois la réponse à « est-ce que ça charge » devenue oui.

Ça change le profil de risque : il n'y a pas besoin de gagner la course pour
que le travail ait de la valeur.

## Règle de méthode, héritée de quatre défauts

Quatre bugs ont été trouvés par **mutation testing**, tous du même motif : une
assertion qui n'exerce jamais le paramètre qu'elle est censée couvrir. Un λ à
zéro qui ne teste aucune crête ; une monotonie non stricte qui accepte un
no-op ; un étage Golay neutralisé et jamais vu ; des tests de *qualité* qui ne
disaient rien du *coût* en bits.

**Avant de déclarer un gate vert : muter le code et vérifier que la suite
échoue.** Et pour les expériences : **une seule variable à la fois**, sur 3
blocs (8 min) avant tout run complet (3,5 h).
