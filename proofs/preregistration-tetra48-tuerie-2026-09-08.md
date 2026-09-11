# Pré-enregistrement — la tuerie précoce de Tetra : le décodage servi contre Planes14, dans un seul processus

**Écrit, commité et TAMPONNÉ le 2026-09-08, AVANT le lancement.** Go de
l'opérateur du 2026-09-08 (« ok go » sur la construction des bras, puis « go
pour le job dans la foulée »).

Ce fichier ne s'édite plus. Ce qui le corrige va dans un `-ECARTS.md`.

**Coût : ~0,02 $** (*estimé*, un job `l40sx1` de quelques secondes de calcul ;
les six bras actuels tournent en 4 s et ont été facturés 0,00 $ le 2026-09-05,
job `6a9c47d5`). Plafond de timeout 20 min, soit **0,60 $ au pire**. Aucun
encodage, aucun artefact. Total projet avant : **134,14 $**.

---

## §1 — La question

Tout le dossier F1 lit son rapport à Planes14 sur **un autre processus**. Le
journal du plancher l'écrit lui-même, `docs/mesures/f1-rang-plancher-2026-09-05.txt:97` :

> « B vient d'un autre processus ; T/B = 1,20 est une **lecture**. F1d met
> Planes14 dans le même processus. »

Et la règle dure 5 interdit de diviser un × entre processus. Donc le `0,61 ×`
que le dossier promène depuis le 2026-09-05 **n'est pas une mesure**.

Deux choses manquent pour qu'elle en devienne une, et ce banc les ajoute
toutes les deux :

1. **Planes14 dans le processus**, sur son propre flux, 14 octets par bloc
   contre 6, adressé à plat comme le noyau servi l'adresse.
2. **Un décodage Tetra réellement servi** : `tv_f1r_v3` ne lit pas le bit de
   gain (`llvq_f1rank.cuh:143`, « Not read here »), ne normalise pas le point,
   et produit son produit scalaire en ordre trio contre une activation en
   ordre naturel. `tv_f1r_v3g` fait les quatre — gain, magnitude, permutation,
   origine — dans la **même boucle de tuiles**, même grille, mêmes deux
   barrières, même mise en scène partagée, même queue, même réduction de warp.

**La question : `t(v3g) / t(planes14)`, formé tour par tour dans un seul
processus.**

## §2 — Ce que ce banc N'EST PAS

- **Ce n'est pas F1d.** F1d tourne sur un vrai `.llvq`, avec les vraies
  étiquettes, dans `planesbench`, contre QTIP et AWQ dans leurs propres
  grilles. Ici les mots sont un flux synthétique et les étiquettes sont
  uniformes.
- **Ce n'est pas un tok/s.** 48 % du temps par token est hors des matmuls
  (*mesuré*, attribution du 2026-08-05), donc un rapport noyau se comprime
  bout en bout. F1e est ce qui mesure le tok/s.
- **Ce n'est pas une mesure de qualité.** `v_proj` en int4 n'est pas dans ce
  banc ; `tv_q4_h` n'a jamais tourné sur une carte.

C'est **une porte**, et une seule : est-ce que ça vaut la peine de continuer.

## §3 — Le protocole, verbatim

```
cargo run --release -p llvq-cuda --bin f1rankfloor
```

Huit bras dans un processus, `LAYERS = 36` copies distinctes par forme, sept
formes du Qwen3-4B, 252 lancements par tour. 14 tours, 2 écartés ; le tour `r`
ouvre avec le bras `r mod 8` ; **toute différence est formée tour par tour**,
jamais comme un quotient de minima (règle 7).

Les deux flux sortent du **même** `f1r_fill`, donc aucun format n'a de
générateur plus aimable que l'autre.

Contrôles, tous avant le premier chronomètre :

| # | quoi |
|---|---|
| 1 | 256 blocs × 7 formes décodés sur carte = `llvq_bench::f1::rank`, chaque coordonnée |
| 8 | 256 blocs × 7 formes **servis** sur carte = `llvq_search::tetra::Tetra::decode` mis à l'échelle par `reconstruct_shape_gain`, chaque coordonnée — le module de production, pas l'étalon de banc, parce que l'étalon parle en ordre trio et n'applique aucune échelle : il ne peut structurellement pas voir la permutation ni la magnitude |
| 2, 3 | chaque sortie finie, écrite, et ≠ `word` ≠ `nullk` |
| 6 | registres et octets locaux des huit noyaux, lus sur la fonction chargée |
| flux | chacun des deux ≥ 4 × la L2, sinon c'est un taux de succès de cache |

## §4 — La règle de décision, posée avant

Le rapport lu est **à tête identique** (règle 4) : le plancher de lancement de
`nullk` est retiré des deux côtés.

```
  R = (t(v3g) − t(nullk)) / (t(planes14) − t(nullk))
```

| lecture | ce qu'on fait |
|---|---|
| **R ≤ 0,80**, borne haute de l'étendue comprise, et `local_bytes = 0` | **VERT.** L'axe continue : transcodeur, câblage, F1d, F1e. |
| **R ≥ 0,90**, borne basse comprise | **ROUGE — l'axe s'arrête.** Le format ne paie pas son noyau, et les 8 $ de F1e ne partent pas. La décision de tuer reste à l'opérateur (règle du 2026-09-05) ; ce banc lui donne le critère, il ne tranche pas seul. |
| entre les deux, ou étendue à cheval | **Rien n'est décidé.** On écrit le chiffre, et l'arbitrage passe à F1d sur un vrai fichier, là où les étiquettes sont réelles. |
| `local_bytes ≠ 0` sur un bras servi | **ROUGE quel que soit R** : un déversement sur le chemin le plus chaud est le défaut que le chemin servi refuse par construction (`fused_cuda.rs:306-311`). |

`num_regs` est un **drapeau, pas une porte**. Le seuil de 40 vient d'une falaise
d'occupation réelle sur sm_89 (granule de 8 registres à 256 fils : 41 registres
font tomber de 6 blocs par SM à 5, `llvq-cuda/src/occ.rs:194`), mais cette perte
est **déjà dans le temps mesuré**. Poser deux portes sur la même chose, c'est
s'autoriser à choisir après coup.

## §5 — Prédiction signée

Écrite avant le lancement, lue contre le résultat quoi qu'il arrive.

| grandeur | prédiction | fourchette |
|---|---|---|
| `nullk` | 2,18 ms | 2,15 – 2,22 |
| `f1r_v3` (rejeu du 09-05) | 3,88 ms | 3,80 – 3,95 |
| `v3g` | **4,05 ms** | 3,95 – 4,25 |
| `v3g − f1r_v3`, le prix de servir | **+0,18 ms** | +0,10 – +0,35 |
| `planes14` en processus | **5,15 ms** | 4,70 – 5,60 |
| `R`, à tête identique | **0,63** | 0,55 – 0,75 |
| `num_regs` de `tv_f1r_v3g` | **44** | 40 – 48 |
| `local_bytes` de `tv_f1r_v3g` | **0** | 0 |

Le raisonnement, pour qu'il soit réfutable : `v3g` ajoute six `__dp4a`, deux
multiplications et une permutation gratuite (indices constants dans les boucles
déroulées) sur ~24 FMA et six lectures par bloc — donc un coût de l'ordre de
10 % du décodage. `planes14` lit **2,33 ×** les octets de Tetra, et le banc de
2026-09-05 mesurait Planes14 − nullk = 2,797 ms ailleurs, ce qui place le bras
autour de 4,98 ms ici ; j'élargis vers le haut parce que ce flux-là est neuf
dans ce processus et que rien ne garantit qu'il se comporte comme celui de
l'autre.

**Je prédis vert.** Si c'est rouge, la prédiction est fausse sur `R` et le
dossier apprend quelque chose qu'aucune lecture transportée ne pouvait lui
dire.

## §6 — Ce qui survit à un rouge

Écrit d'avance pour qu'un rouge ne soit pas une perte sèche :

- le **`B` en processus**, un chiffre que `docs/format-noyau.md` §6 n'a jamais
  eu et qui vaut pour tout format futur ;
- `llvq_tetra48.cuh` et ses cinq tests, qui sont le décodage servi de
  **n'importe quel** format qui ne déplie pas ;
- le contrôle 8, qui relie pour la première fois une sortie carte au module de
  production `llvq_search::tetra` plutôt qu'à l'étalon de banc.

## §7 — Limite de portée

Le 4B seul. `docs/ETAT.md` §5 septies a déjà mesuré que l'écart de Tetra double
du 4B au 8B sur les deux axes de qualité ; rien ici ne dit quoi que ce soit du
8B, et un vert au 4B ne se transporte pas.
