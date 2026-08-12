# Passation — lot X, la partie Mac (2026-08-12)

> Ce que le runbook [`runbook-lot-x-mac.md`](runbook-lot-x-mac.md) demandait de
> lancer sur le Mac de dev, lancé. **0 $, aucun téléchargement.** Trois
> verdicts, dont un rouge qui enterre un chantier et deux verts qui rendent le
> banc CUDA de la semaine suivante purement une question de vitesse.
>
> Spec : [`spec-memoire-extreme-2026-08-12.md`](spec-memoire-extreme-2026-08-12.md).
> Mesures brutes : [`mesures/radixstudy-x4-2026-08-12.txt`](../mesures/radixstudy-x4-2026-08-12.txt),
> [`mesures/rtbits-e1c-4b-2026-08-12.txt`](../mesures/rtbits-e1c-4b-2026-08-12.txt),
> [`mesures/e1c-sweep-4b-2026-08-12.txt`](../mesures/e1c-sweep-4b-2026-08-12.txt).

## Tableau de bord

| étape du runbook | verdict |
|---|---|
| 1. boucle rapide + clippy | ✅ 11 tests E1c verts, **zéro warning** clippy sur tout le workspace |
| 2. **X4 — l'étude E3** | 🔴 **ROUGE sur les deux bras**, dont celui qui fait foi |
| 3. compte de bits E1c sur le fichier réel | ✅ les quatre lignes du runbook **au chiffre près** |
| 4. sweep intégral, 150 681 600 blocs | ✅ **vert en 459 s**, les deux variantes exactes |
| 5. histogramme de routage MoE | ⚠️ **le risque caché est réel** — 31,4 % des cellules sous le rang plein |
| 6. devis | ✅ recalé, et **plus bas qu'annoncé** — voir plus bas |

## X4 — E3 est enterré sur papier, pour 0 $

`bin/radixstudy`, deux bras. Le critère d'ouverture était posé **avant** la
mesure (spec §X4) : ≤ 2,6 b/poids noyau projeté **et** un décodeur à
profondeur bornée sans état sériel inter-slot.

| variante | shift-only | b/p moy | b/p groupé 32 |
|---|---|---|---|
| archive (le fichier) | non — ~509 ops sérielles | 2,1498 | 2,1912 |
| radix2 | non — rangs de multiensemble sériels | 2,3102 | 2,5066 |
| radix2 + golay 12 b | non, idem | 2,3257 | 2,5101 |
| **`golay_tight`** | **oui, profondeur 24** | **2,9034** | **3,0444** |
| `golay70` (mesuré, écarté) | oui, profondeur 24 | 3,0621 | 3,1866 |
| `perslot` (= `Planes`) | oui, profondeur 24 | 2,9556 | 3,2060 |

Pondéré par les **blocs réels** du 4B scellé — les 150 681 600, pas les 383
classes à poids égal. **Le meilleur point shift-only est à 3,0444, soit 17 %
au-dessus du seuil**, et 2,9034 même dans la comptabilité la plus généreuse
(moyenne, sans le groupement de 32). Le bras à poids égal sort rouge aussi,
plus largement (3,1945, +23 %).

**La ligne qui explique pourquoi** : sur les blocs réels, le point *dans* sa
classe coûte déjà 41,50 bits de moyenne (⌈log₂|classe|⌉) sur les 47 de
l'index — il ne reste que ~5,5 bits pour le choix de classe, que **toute**
variante à champ de classe explicite repaie en 10 bits d'en-tête. La
décomposition shift-only ne peut pas être moins chère que ce qu'elle
explicite.

> ⚠️ **Une divergence de prose à corriger, sans effet sur le verdict.** Le
> runbook annonçait « 2,73 contre un critère de 2,60 » sur le bras à poids
> égal : c'est la colonne **b/p moy** de `golay_tight`. Le bin, lui, juge sur
> la colonne **groupé 32** (3,1945). Les deux comptabilités sont rouges sur
> les deux bras, donc le verdict est robuste au choix — mais les deux textes
> ne citaient pas le même nombre, et c'est exactement le motif que le dossier
> se reproche ailleurs.

**Ce que ça ferme.** Le chantier E3 (le 24-32 Go), et avec lui la ligne
« K2.6 sur un poste » de l'étude MoE. **Le plafond mémoire du projet est
E1c.** À consigner comme un verdict, pas comme un échec : E2 s'était enterré
au banc pour ~0,2 $, E3 s'enterre sur papier pour 0 $, et dans les deux cas
le critère avait été écrit d'avance.

## Le compte de bits E1c — le runbook au chiffre près

`bin/rtbits ~/llvq-q4b.llvq`. Attendu et obtenu, identiques :

| layout | b/poids payload | b/poids noyau | Go de stream |
|---|---|---|---|
| `Slot32` | 5,3756 | 5,5096 | 2,50 |
| `Planes14` — ce qui tourne | 4,6667 | 4,8040 | 2,18 |
| **`E1c14`** | **4,4167** | **4,5551** | 2,07 |
| `Planes12x` | 4,2029 | 4,3424 | 1,97 |
| **`E1c12`** | **3,6196** | **3,7618** | **1,71** |

Et le fait qui porte le layout : le terme d'exceptions d'`E1c12` (0,2029)
est **le même** que celui de `Planes12x`, à la table près — donc l'écart
entre les deux layouts **est** le bourrage, 14 bits par bloc, et rien
d'autre. Idem à 6 bits entre `E1c14` et `Planes14`.

Les trois tests d'acceptation de `rtbits` épinglent 4,5551 et 3,7618 à
5·10⁻⁴ près : le bin et la suite ne peuvent plus diverger en silence.

## Le sweep intégral — la preuve d'exactitude, acquise

`cargo test --release -p llvq-artifact --test e1c_format -- --include-ignored`.
**150 681 600 blocs, 459 s**, les deux variantes contre le décodeur d'archive
et le flux principal d'`E1c12` contre celui de `Planes12x` — le même standard
auquel `Planes14`, `Planes12x` et `Golay70` ont chacun été tenus.

```
E1c sur 150681600 blocs — payload b/poids :
  Planes14 4.6667 → E1c14 4.4167  (-0.2500)
  Planes12x 4.2029 → E1c12 3.6196  (-0.5833)
```

Les taux sortis du sweep sont **les octets réellement construits**, pas une
formule ré-appliquée : ils tombent au chiffre sur ceux de `rtbits`, qui les
dérive du compte de blocs. Deux chemins indépendants, même nombre.

**Ce que ce vert débloque, et ce qu'il ne débloque pas.** Le bras de banc CUDA
(X3) devient un **pur** test de vitesse à ~0,2 $ : plus aucune question de
correction ne reste à y mélanger. Il ne dit **rien** de la vitesse — le noyau
lirait 82 ou 106 mots par groupe là où `Planes14` en lit 4-5 par lane, et un
compte niveau source a déjà été faux d'un facteur 2 sur ce noyau.

## Le routage MoE — le risque caché est réel, et il est chiffré

`ops/moe_routing.py` (écrit pour ce lot), `gpt-oss-20b` MXFP4 déquantifié en
bf16, **131 072 tokens de C4**, MPS, 438 s. Hors de notre pipeline
délibérément : n'importe quel runtime donne les décisions du routeur, et les
prendre chez `transformers` évite de confondre un défaut de routage avec un
défaut de notre passe avant. Journal :
[`../mesures/moe-routing-gptoss20b-2026-08-12.txt`](../mesures/moe-routing-gptoss20b-2026-08-12.txt),
comptes bruts [`../data/moe-routing-gptoss20b-2026-08-12.json`](../data/moe-routing-gptoss20b-2026-08-12.json).

⚠️ **L'unité est la cellule `(couche, expert)`, pas l'expert.** Chaque expert
de chaque couche a sa propre hessienne ; agréger sur les couches donne une
distribution bien plus plate que la réalité (Gini 0,169 agrégé contre
**0,59–0,77 par couche**) et c'est le piège de ce tableau. 24 couches × 32
experts = **768 cellules**.

| | |
|---|---|
| charge uniforme | 16 384 routages/cellule |
| cellules **mortes** (zéro routage) | **1** — `L15/e20` |
| cellules **sous 2 880** (le rang de la hessienne) | **241, soit 31,4 %** |
| quantiles de charge | p1 = 17 · p10 = 240 · médiane = 7 916 · p100 = 117 450 |

**Le volume qu'il faudrait**, à 2 880 routages par cellule :

| couverture visée | tokens de calibration | facteur |
|---|---|---|
| 50 % des cellules | 47 690 | ×0,36 — déjà acquis |
| 75 % | 212 999 | ×1,6 |
| **90 %** | **1 572 209** | **×12** |
| 95 % | 3 411 545 | ×26 |
| 99 % | 21 769 744 | ×166 |
| 100 % | impossible — l'expert mort ne s'achète pas |

**Trois lectures, dans l'ordre d'utilité.**

1. **Le runbook avait raison de s'inquiéter, mais pas sur le bon axe.** Notre
   profil par phase dit que l'encodeur pèse 90 % du run et qu'il est
   proportionnel aux **poids**, pas aux tokens ; les passes avant pèsent 1,2 %
   à 8B. Un ×12 de calibration ne multiplie donc que ce 1,2 % — soit **~+13 %
   de run**, pas une explosion. C'est ×166 qui ferait basculer la note, et
   ×166 n'est pas nécessaire pour un premier point.
2. **L'expert mort n'est pas un problème de volume**, et c'est le fait le plus
   dur du lot : aucun corpus ne le ressuscite. Il faudra une politique
   explicite (le laisser en pleine précision, ou le quantifier sur une
   hessienne régularisée), et cette décision n'existe nulle part dans notre
   pipeline aujourd'hui — qui suppose partout une hessienne inversible.
3. **La mesure est un plancher de difficulté, pas un plafond.** `gpt-oss`
   active 4/32 experts, soit **12,5 %**. Les cibles réelles sont bien plus
   creuses — `Qwen3-30B-A3B` 8/128 = 6,3 %, K2.6 8/384 = 2,1 % — donc leur
   déséquilibre sera **pire**. Extrapoler ce tableau au 30B-A3B serait
   exactement le raccourci que ce dossier refuse ; ce qu'il autorise, c'est de
   dire que le problème existe et qu'il faut le mesurer sur la cible avant de
   payer.

> 🕳️ **Un artefact de sortie corrigé sur place, parce qu'il est instructif.**
> La première version du script résumait par « l'expert le moins servi », donc
> divisait par une cellule à zéro : elle annonçait « il faudrait
> 2 880 000 000 000 000 tokens », un nombre vrai et parfaitement inutile qui
> écrasait la vraie question. Le pire n'est pas le bon résumé d'une
> distribution à queue — les quantiles le sont. Le mode `--from-json` rejoue
> le verdict sans recharger le modèle, donc la correction n'a pas coûté un
> second run.

## Le devis, recalé

`uv run ops/run.py selftest` puis `estimate Qwen/Qwen3-30B-A3B --dtype bf16` :

| | |
|---|---|
| poids quantifiés | 2,72 Md |
| poids portés en 16 b | 0,62 Md — **18,6 % des poids, 63 % de l'artefact** |
| artefact projeté | 2,0 Go (×3,4) |
| encodage Leech | 21,4 cœur-h |
| **`rtx-pro-6000`** | **~5,74 $** |

⚠️ **Deux réserves, et elles jouent en sens opposés.**

1. **L'estimateur est bas, et il le dit lui-même** : sur le run 4B réel, son
   selftest ne rend compte que de **59 %** du temps facturé (8 588 s d'encodage
   + 19 s de Cholesky contre 14 447 s mesurés). Au même ratio, le run MoE
   coûterait plutôt **~10 $** que 5,74. C'est le même défaut que le 25 % de bas
   du dé-risquage 32B, mesuré cette fois au lieu d'être subi.
2. **Ces ~10 $ ne sont qu'un bras.** Le gate X5-MoE annoncé à 25-55 $ compare
   le 30B-A3B au **32B dense**, et il y ajoute les évaluations. Le devis
   ci-dessus ne périme donc pas l'annonce : il en chiffre la moitié la moins
   chère.

Note utile pour la suite : les 63 % d'artefact portés en 16 bits sur ce MoE
sont le même piège d'embedding qu'au 8B, en pire. Un ratio de compression
annoncé sans son mécanisme y serait encore plus trompeur.

## Ce qui reste, et ce qui bloque

| item | état |
|---|---|
| **X3** — bras de banc CUDA (K-U / K-S) et noyau `e1c.cu` | ❌ pas écrit. Le sweep vert le rend légitime : c'est un pur test de vitesse à ~0,2 $. Critères déjà posés : ≥ 1,9× pour remplacer `Planes12x`, ≥ 2,05× pour remplacer `Planes14`, sous 1,6× l'échelle se referme côté transposition |
| **X4** | 🔴 clos |
| **politique « expert mort »** | ❗ **nouveau, et bloquant pour tout run MoE** : notre pipeline suppose partout une hessienne inversible. Il en existe au moins une qui ne le sera jamais. À trancher avant le gate X5-MoE, pas pendant |

⚠️ **Rien de ce lot ne mesure une vitesse.** Le code livré compte des bits et
prouve une bijection. « Le transposé va-t-il aussi vite » reste entier et ne
se tranche que sur carte — un compte niveau source a déjà été faux d'un
facteur 2 sur ce noyau.

> 🚫 **Et c'est pourquoi `data/echelle-formats.csv` n'a PAS reçu de ligne
> `E1c`, délibérément.** Le schéma de ce CSV est celui d'une mesure de vitesse
> — `med_ms`, `gbps`, `pct_byte_bound`, `ratio_vs_fp16` et sa plage — et il
> alimente une table du papier que `paper/scripts/check_tables.py` vérifie.
> Y poser une ligne dont tout sauf `bpw_kernel` serait vide (ou pire,
> extrapolé) transformerait un compte de bits en point de courbe débit↔taux.
> **La ligne s'ajoutera quand X3 l'aura remplie, pas avant.** À la prochaine
> session qui trouvera ce CSV « incomplet » : c'est voulu.

## Dette de branche — soldée le même jour

La branche `claude/html-mini-cours-planes-w9prvm` avait été tirée **avant** la
refonte documentaire du 2026-08-12 : ni `HISTORIQUE.md`, ni `PLAN.md`, ni
`archive/`, et 17 commits du lot Golay70 v2 absents. Réglé :

- **Fusion** de `claude/golay70-memory-performance-ksbbzs` — propre, aucun
  conflit. Les deux seuls fichiers touchés des deux côtés (`runtime.rs`,
  `rtbits.rs`) ne divergeaient que par des chemins **en commentaire**.
- **Cinq renvois périmés réécrits** vers `docs/archive/` :
  `e1c.rs`, `e1c_format.rs`, `radixstudy.rs`, `rtbits.rs`,
  `cours-layouts-runtime.html` et la spec elle-même.
- **Quatre documents de session rangés** dans `archive/` selon la convention
  de la refonte (spec, runbook, étude MoE, cette passation), leurs liens
  relatifs réécrits et **vérifiés un par un**.

⚠️ **Une surface reste en retard, et c'est délibéré** :
[`cours-layouts-runtime.html`](../cours-layouts-runtime.html) enseigne
l'échelle des formats **sans `E1c`** — il a été écrit le matin, avant que les
deux barreaux existent. À reprendre quand X3 aura tranché : y ajouter un
layout dont la vitesse est inconnue apprendrait la mauvaise leçon, et ce
dossier a déjà payé pour avoir publié un compte de bits comme s'il disait
quelque chose du débit.
