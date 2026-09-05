# Pré-enregistrement — le plancher compilé du décodeur F1 à table universelle (`tv_f1r`)

**Écrit le 2026-09-05, commité et TAMPONNÉ avant la première milliseconde.**
Vague 2, plafond **2,00 $**, dépensé 0,02 $ ; ce job coûte **~0,01 $**, plafond propre **0,10 $** (go opérateur du 2026-09-05).
Code mesuré : commit `bc543ba` (branche `f1/plancher-table`).

🚨 **Ce fichier ne s'édite plus.** Ce qui le corrige va dans un `-ECARTS.md`.

## §1 — Ce n'est pas une porte, et ce banc ne tue rien

Règle d'opérateur du 2026-09-04 (`docs/METHODE.md` §1) : une porte se pose sur un critère
fondamental. Règle du 2026-09-05, même section : **un kill s'arbitre sur un critère fondamental et
par l'opérateur seul** ; un plancher informe, il ne prononce pas. Le plancher de table du 05 a été
lu comme un kill le matin même et retiré le soir
(`proofs/preregistration-f1-plancher-table-2026-09-04-ECARTS.md` §É6). Celui-ci ne le sera pas.

Ce qu'il informe : **sous quelle forme écrire `tv_l3e8`** (F1d), et s'il faut réécrire le décodeur
avant. Un choix d'ingénierie, pas un verdict.

## §2 — La question

F1 avec la table universelle : une seule table de 4 096 lignes × 4 o = **16 Kio** pour les 528
régions, les octets de motif venant de trois petites tables (128 + 128 + 2 048 o), le point rendu
par ~200 à 300 opérations entières par bloc. Sa qualité est mesurée : −0,61 et −0,64 pp de rétention
contre F1 exact, 88,88 % contre 89,48 sur 4 000 blocs, témoin boule-12 à 92,00 (*mesuré*,
`llvq-bench/examples/f1rankbench.rs`). Le plancher de table du 05 dit qu'une table de 16 Kio
coûte 0,34 à 0,66 ms dans notre géométrie (*mesuré*, `docs/mesures/f1-plancher-table-2026-09-05.txt`).

**Ce que personne n'a mesuré : le décodage arithmétique compilé.** C'est là qu'E1v est mort — 79
registres, 0,25× f16 (`docs/mesures/e1v-cuda-2026-08-16.txt`). La question est donc : lire le mot de
6 octets, lire trois lignes de la table et trois octets de motif, dérouler les 24 coordonnées et
les multiplier — combien, en registres, en octets locaux, et en millisecondes sur les 252 lancements ?

## §3 — Le montage

`bin/f1rankfloor` (`llvq-cuda/src/bin/f1rankfloor.rs`), noyaux `llvq-cuda/kernels/f1rank.cu` et
`llvq_f1rank.cuh`, dans la géométrie de `tv_nullk` : 7 formes du 4B × 36 couches = 252 lancements
par tour, un warp par ligne, 256 threads, tuile de 128 blocs. Trois bras, tous à chaque tour,
**ordre tournant** (le bras qui ouvre le tour change à chaque tour — le Didx négatif du 05 était
peut-être un effet de position), 11 tours dont 2 de chauffe écartés — 9 tours gardés, chaque bras
à chaque position exactement trois fois —, différences formées tour par tour :

```
  nullk   la même passe sans un octet de poids                (le plancher, dans CE processus)
  word    nullk + lecture du mot de 6 o par bloc, replié en un flottant, sans décodage
  f1r     word + décodage complet : 3 lignes de table, 3 octets de motif, 24 coordonnées, 24 FMA

  S  = t(word) − t(nullk)    le flux F1 dans notre géométrie
  Du = t(f1r)  − t(word)     table + décodage arithmétique
  T  = t(f1r)  − t(nullk)    ce que F1 dépense en flux ET décodage
```

Le flux : **36 copies distinctes** du flux de mots, pseudo-aléatoires, engendrées sur la carte,
6 octets par bloc et un pas de ligne arrondi à 8 octets : **0,908 Go** en tout, 9,0 × la L2 de
96 Mio (le 0,98 Go du disque compte la queue en f32 et les échelles, que ce banc porte à part) —
une copie par forme tiendrait dans la L2 dès la deuxième couche et `S` mesurerait la L2, pas la DRAM. Tout mot de 48 bits tiré uniformément est une étiquette valide
(chaque champ est une puissance de deux), donc les lignes de table sont tirées uniformément sur les
16 Kio : légèrement **pessimiste** contre l'accès réel, qui concentre ~70 % des lectures sur ~6 Kio
(`f1rankbench`, déciles d'index). Le bit de gain est ignoré : ce banc n'applique pas l'échelle du
noyau servi.

Commande du job (image reconstruite par `ops/run.py publish` sur le commit ci-dessus) :

```
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv | tee $OUT/gpu.txt
f1rankfloor 2>&1 | tee $OUT/f1rankfloor.txt
```

## §4 — L'échelle de lecture

`T` se lit contre **B = t(Planes14) − t(nullk) = 5,103 − 2,306 = 2,797 ms** (*mesuré*,
`docs/format-noyau.md` §6, autre processus). C'est une comparaison entre processus ; `format-noyau`
§6 interdit de *soustraire* un temps d'un autre processus, pas de lire une différence contre une
différence, et `nullk` s'est reproduit d'un processus à l'autre à −5 % (2,191 contre 2,306). Le
rapport `t(f1r)/t(nullk)` contre `5,103/2,306 = 2,21` est publié à côté, avec la même réserve.
F1d, plus tard, mettra Planes14 dans le même processus.

`Du` se lit contre le plancher de table : `D(16 Kio) = 0,663 ms` du 05. `Du − 0,663` est le prix
de l'arithmétique et des trois petites lectures, à ±0,1 ms — la résolution de ces bancs
(É7 du plancher).

## §5 — Les contrôles, et si l'un tombe aucun chiffre n'est publié

1. **Le décodeur de la carte rend les mêmes points que la référence Rust.** Pour chacune des 252
   copies (36 couches × 7 formes), les 256 premiers blocs du flux en ordre ligne-majeure — donc
   plusieurs lignes, ce qui exerce aussi le pas de ligne — sont décodés sur la carte (`tv_f1r_dump`)
   et comparés coordonnée par coordonnée à `llvq_bench::f1::rank::decode_word`, sur le même flux (le
   mélangeur qui engendre les mots est écrit une fois dans le `.cuh` et miroité en Rust) : 64 512
   blocs. Une différence : sortie 1, aucun temps.
   Avant la carte : le même `.cuh` compilé par clang++ sur le Mac rend les mêmes 24 coordonnées que
   Rust sur 10 000 mots (`llvq-cuda/tests/f1rank_matches_rust.rs`), et 2 000 mots décodés tombent
   dans Λ₂₄ sous `llvq_core::Leech::contains`.
2. **Rien n'est élidé** : la sortie de `f1r` diffère de celle de `word` ET de celle de `nullk` ; la
   sortie de `word` diffère de celle de `nullk`.
3. **Tout est observable** : chaque ligne de sortie écrite, finie, pas toute nulle.
4. **Le flux ne tient pas en L2** : octets de mots ≥ 4 × L2, vérifié contre l'attribut de la carte
   (9,0 × attendu ; le point DRAM du plancher du 05 exigeait 20 ×, ici `S` n'est pas le nombre qui
   décide et la réserve est portée au §8).
5. **Un seul processus, une seule géométrie**, celle de `nullk`.
6. **Registres et octets locaux** de `tv_nullk`, `tv_f1r_word`, `tv_f1r` imprimés depuis les
   attributs de fonction.

## §6 — Ce qui est publié, et ce qui ne se compare pas

Publié : `S`, `Du`, `T` en médiane [plage] ; `t(f1r)/t(nullk)` ; registres et locaux des trois
noyaux ; les contrôles.
Ne se compare pas : à QTIP (autre grille, règle 5) ; à Planes14 par soustraction (§4) ; à un tok/s
(ce banc est 51 % d'un token, et sans l'échelle du noyau servi).

## §7 — Ce que le résultat décide, et ce qu'il ne décide pas

Ce n'est pas une porte ; la table dit ce qui s'écrit ensuite.

| résultat | suite |
|---|---|
| contrôle 1 échoue | aucun chiffre ; un bug de décodeur, à corriger avant tout |
| T ≤ 2,797 ms, registres ≤ 64, locaux = 0 | F1d s'écrit avec CE décodeur ; l'encodeur de production et le format v2 continuent |
| 2,797 < T ≤ 3,50 ms, registres ≤ 64, locaux = 0 | F1d s'écrit, avec la projection publiée : passe +0 à +0,7 ms, soit 100,6 → ~94 tok/s au 4B (*estimé*) ; l'opérateur pèse contre la VRAM et la classe |
| T > 3,50 ms, ou registres > 64, ou locaux > 0 | le décodeur est réécrit avant F1d (l'arithmétique ou les registres, pas la table) ; l'opérateur décide si la semaine se dépense |
| autrement | non tranché, décision d'opérateur |

Aucune ligne ne s'appelle « F1 est mort ».

## §8 — Prédiction signée, opposable

**S entre 0,9 et 1,3 ms** (0,908 Go à 700–980 Go/s, la fourchette des débits nets mesurés de ce
dépôt ; à 9 × la L2 et non 20 ×, une part de l'ordre de 10 % des lectures peut être servie par la L2,
ce qui tire S vers le bas de la fourchette). **Du entre 0,9 et 1,6 ms** (0,663 de table à accès uniforme, plus 0,2 à 0,9 ms
d'arithmétique : ~250 opérations par bloc, non masquées à 0,69 ms, masquées à ~0,2 comme les
~30 de `hash3`). **T entre 2,0 et 3,0 ms, valeur centrale 2,5.** Registres 40 à 64, locaux 0.

Motif : le décodage est branch-free, 24 sélections et 24 conversions entier → flottant sur 8
coordonnées par ligne de table ; c'est plus que les ~250 de Planes14 mais du même ordre, et il n'a
ni boucle sérielle ni tableau indexé dynamiquement.

**Ce qui la rendrait fausse de façon instructive** : T sous 1,8 ms — l'arithmétique serait
gratuite et le flux F1 se recouvrirait avec la table, ce qui changerait la lecture de tout le
budget H du 04 ; ou T au-dessus de 3,5 ms avec des registres sous 64 — les trois petites lectures
d'octets de motif (dépendantes : `s8 → br → s16 → c3`) ne seraient pas masquées, et le décodeur
devrait lire ses motifs autrement.

⚠️ Historique des prédictions signées de ce dossier : deux fausses le 08-25 ; une juste sur le
nombre et fausse sur la conclusion le 09-02 ; une réfutée sur un tirage et confirmée sur l'autre
le 09-04 matin ; deux justes le 09-04 ; le 09-05 : juste sur D(4 Mio) et D(16 Kio), fausse sur
Dsm ; et la prédiction de perte de la table universelle (2,0 à 4,5 pp) fausse dans le bon sens
(0,6 mesuré). Celle-ci est opposable, pas crédible d'avance.

## §9 — Ce que ce banc ne peut pas être

Un coût de production : pas d'échelle de gain, des étiquettes uniformes et non celles de vrais
poids, et pas de Planes14 dans le processus. Une mesure de qualité : rien ici ne touche à un modèle.
Ce qu'il est : le premier décodeur F1 **compilé et vérifié sur la carte contre une référence**, ce
que le plancher du 05 n'était pas.
