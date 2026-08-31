# Format de noyau — mesures, impasses, architecture (2026-07-31)

> 🗓️ **BANDEAU D'ÉTAT — dernière revue intégrale le 2026-08-08 ; revue
> partielle du 2026-08-16 ci-dessous, amendements ponctuels datés dans le
> corps jusqu'au 2026-08-17.** Ce document décrit
> l'état **Metal / `Slot32`** du noyau, qui n'est plus l'état de référence.
> Trois choses ont changé depuis, toutes mesurées :
>
> 1. **`Slot32` a été remplacé.** Le layout de production est **`Planes14`** —
>    plans de bits binaires au lieu des masques one-hot, stride uniforme 14 o,
>    plus de table de bases : **4,804 b/poids contre 5,510, et 1,14× plus
>    rapide à contenu décodé identique**
>    ([`mesures/c1-planesbench-2026-08-06.txt`](mesures/c1-planesbench-2026-08-06.txt)).
>    Deux points de fonctionnement supplémentaires existent : `Planes12x`
>    (4,342 b/poids, overlay exact, **mesuré non branché**) et `Golay70`
>    (3,589 b/poids, **1,31× — mesuré et écarté**, sous le critère de 1,6×).
> 2. **Le noyau est branché.** Voir la correction en fin de section « Ce que ce
>    chiffre ne couvre pas ».
> 3. **Le 2,07× se publie en plage.** Trois invocations du banc non modifié
>    rendent 2,029× · 2,050× · 2,080× à octets et erreurs identiques
>    ([`mesures/thesis-temoin-2026-08-04.txt`](mesures/thesis-temoin-2026-08-04.txt)).
>
> La note de provenance des trois comptabilités RAM, en fin de fichier, reste
> valable et reste la référence du dépôt sur ce point.

> 🗓️ **REVUE DU 2026-08-16 — un fait neuf recadre tout ce document.** Rien
> ci-dessous n'est retiré : les comptabilités tiennent, les ns/bloc tiennent.
> Ce qui change est le **dénominateur** contre lequel un gain de format se lit.
>
> 1. 🆕 **LE PLANCHER EST MESURÉ, et il vaut 45,2 % du bras servi.** Une passe
>    de projections qui ne lit **aucun poids** coûte **2,305 ms** contre
>    **5,102** pour `Planes14`, là où `Planes14` est déjà à **2,16×**. Section
>    « Le plancher », en fin de fichier — c'est le nouveau fait dominant sur le
>    coût du décodage, et il vaut plus que toutes les lignes de ce document
>    prises ensemble.
>    🕳️ **Ce point a porté « donc TOUT TRAVAIL DE FORMAT PLAFONNE À 4,77× FP16 »
>    jusqu'au 2026-08-21, et c'est mesuré faux.** Le fait ne bouge pas — les
>    2,305 ms sont les mêmes, rejoués à 2,306 dans le banc à dix bras ; c'est
>    l'**interprétation** qui tombe. Un plafond de format suppose qu'aucun noyau
>    ne passe sous `nullk` ; **le noyau QTIP y passe**, dans le même processus et
>    sur les mêmes formes : **2,246 ms contre 2,306** en lisant 0,91 Go, soit une
>    séparation de 2,7 % contre une résolution de 0,72 %
>    ([`mesures/f2-p3-qtip-banc-2026-08-21.txt`](mesures/f2-p3-qtip-banc-2026-08-21.txt)).
>    Le mécanisme est nommé et il n'est pas une erreur de mesure : `nullk`
>    partage **notre** géométrie de lancement — un warp par ligne de sortie, 252
>    lancements — et QTIP est lancé dans **la sienne**. Formulation de
>    remplacement, à utiliser partout : **`nullk` est le plancher de notre
>    géométrie de lancement**, pas un plancher machine. Section « Le banc à dix
>    bras ».
> 2. ❌ **Quatre routes sous `Planes14` ont été tentées, toutes bornées en
>    CALCUL, aucune en octets** — E3, `Golay70` v2, `e1c14`, **E1v** (0,25× FP16
>    sur carte, 2026-08-16). Table « Les quatre routes », même section. Le point
>    3 de la section « Trois choses que cette échelle établit » (`CLAUDE.md` §3)
>    disait « la courbe finit par se retourner » ; le plancher dit **pourquoi**,
>    et il dit surtout que c'était le mauvais front.
> 3. ❔ **`e1c12` survit à l'alignement warp** — 4,2880 contre 4,3424 b/poids
>    noyau pour `Planes12x`, soit **−1,3 %** — donc sa question **cesse d'être
>    une question de bits** et devient une question de **vitesse de
>    transposition**. Aucune nanoseconde ne l'a mesurée. Son jumeau `e1c14`, lui,
>    est enterré **au 4B** : +9,0 % une fois aligné.
>    🚨 **Sans « au 4B », cette phrase est fausse — corrigé le 2026-08-17.** La
>    pénalité d'alignement warp vaut **+15,47 % de blocs sur les formes du 4B**
>    mais **+4,18 % sur celles du 14B**, dont les lignes sont plus longues (213
>    et 725 blocs contre 106/170/405). Sur les blocs réels du 14B, `e1c14`
>    aligné rend **4,6410 contre 4,7063 pour `Planes14`, soit −1,4 %** — il
>    passe **sous** ce qu'il remplace
>    ([`mesures/rtbits-14b-2026-08-17.txt`](mesures/rtbits-14b-2026-08-17.txt)).
>    ⚠️ **Cela ne le ressuscite pas** : aucun de ces nombres n'est une vitesse,
>    et `e1c` n'a jamais été dispatché par un banc, à aucune largeur. Ce qui est
>    établi est étroit — **la pénalité d'alignement est une fonction des FORMES,
>    pas une constante du layout.**
> 4. ⚠️ **Le bandeau du 08-08 ci-dessus dit `Planes12x` « mesuré non branché »**
>    — il est **câblé** dans le modèle depuis le 2026-08-09
>    (`LLVQ_FUSED_LAYOUT=planes12x`, `llvq-llm/src/fused.rs`), et il n'est
>    toujours **pas servi** par défaut. « Câblé » n'est ni « servi » ni « non
>    branché » : trois états, et ce document n'en nommait que deux.
>    🆕 **Périmé par G3 le 2026-08-23 : `Planes12x` a été SERVI bout-en-bout au
>    4B.** Ce n'est plus un troisième état supposé mais un run —
>    **85,0 tok/s [84,7–85,1] dans 2,36 Go de carte, ÷3,41 de mémoire, ×1,96 sur
>    le bras dense**, tokens gloutons identiques, divergence au token 89 comme
>    `Planes14`
>    ([`mesures/g-horloges-planes12x-2026-08-23.txt`](mesures/g-horloges-planes12x-2026-08-23.txt),
>    job `6a8c2355…`, 0,79 $). C'est **le point servi le plus compact mesuré** du
>    dépôt. ⚠️ La nomenclature ne change pas pour autant : le **défaut** reste
>    `Planes14`, donc `Planes12x` est aujourd'hui « **mesuré servi, non
>    défaut** » — un quatrième état, et il faut le nommer plutôt que de le
>    replier sur « servi ».

> 🗓️ **REVUE DU 2026-08-25 — deux faits neufs recadrent les deux bornes de ce
> document, et un troisième change ce que « servi » veut dire.** Comme la revue
> précédente : rien n'est retiré, les millisecondes tiennent, les comptabilités
> tiennent. Ce qui change est ce qu'on a le droit d'en conclure.
>
> 1. 🚨 **Le plancher `nullk` n'est pas un plancher machine, et le « plafond de
>    4,77× » qui en dérivait est mesuré faux.** Le noyau QTIP porté dans notre
>    propre banc finit les mêmes 252 projections en **2,246 ms** contre
>    **2,306 ms** pour notre passe qui ne lit **aucun octet de poids** — même
>    processus, mêmes formes, 7 rounds dont 2 jetés
>    ([`mesures/f2-p3-qtip-banc-2026-08-21.txt`](mesures/f2-p3-qtip-banc-2026-08-21.txt),
>    0,89 $). Section « Le banc à dix bras », et l'erratum au pré-enregistrement
>    tamponné y est consigné mot pour mot.
> 2. 🚨 **Tout « × vs FP16 » de ce document est un résultat L40S/Ada.** Sur
>    **A100-SXM4-80GB**, dans le même banc et le même code, **aucun bras à
>    décodage ne bat le FP16** — `Planes14` **0,79×**, `Slot32` 0,73×,
>    `Planes12x` 0,73×, `Golay70` v2 0,62×, v1 0,44×
>    ([`mesures/f4-a100-2026-08-18.txt`](mesures/f4-a100-2026-08-18.txt)). Le lot
>    G en donne le mécanisme, horloges **lues** pendant le banc : 2 520 contre
>    1 410 MHz, épinglées au boost, aucun bridage. Section « L'échelle ne
>    transfère pas ».
> 3. ✅ **`Planes12x` est SERVI, mesuré, au 4B** — 85,0 tok/s dans 2,36 Go
>    (G3, point 4 ci-dessus). Le layout le plus compact du dépôt a quitté l'état
>    « câblé ».
> 4. ✅ **Le poste que ce document déclarait intouché a été entamé** — pas par un
>    format, par la **géométrie**. `D1` fusionne q+k+v et gate+up sur le chemin
>    servi : **252 → 144 lancements de matvec par token**, **×1,061
>    [1,050–1,069]** bout-en-bout à `ROT_SHARE` constant
>    ([`mesures/d1-fusion-servie-2026-08-24.txt`](mesures/d1-fusion-servie-2026-08-24.txt),
>    0,24 $). La garde n° 4 de la section « Le plancher » disait « rien sur
>    `k > 1` … personne ne l'a chiffré » : quelqu'un l'a fait, par une autre
>    porte.
>
> ⚠️ **Ce que cette revue ne touche pas.** Les b/poids, les octets lus et les
> pires erreurs de tout ce document sont inchangés — ce sont les grandeurs
> exactes, elles se reproduisent au chiffre. Et le **papier** est soumis à ACM
> TACO depuis le 2026-08-24 (TACO-2026-428, commit `e21a8bb`) avec le bras QTIP
> intégré à son corps : ce fichier-ci est en retard sur lui, pas l'inverse.

> Branche `g6-format-noyau`. Tout chiffre ci-dessous est mesuré par un banc
> reproductible (`decbench`, `decprofile`, `classprofile`, `arrbits`, `rtbits`,
> `decfast`, `decfull`, `matvec`, `thesis`), et l'ensemble a été passé au crible
> d'un audit adversarial de 5 agents (lecture seule) dont les corrections sont
> intégrées.
>
> ⚠️ **Trois chiffres de ce document ont été corrigés le 2026-08-05 par le lot
> K-1** (mesures locales, journaux dans `docs/mesures/`) : le « ~4,4 b/poids »
> du plafond L ≤ 4, l'échelle bits↔vitesse qui mélangeait trois protocoles, et
> la cause attribuée à l'écart 5,375 / 5,51. Les anciennes valeurs sont
> conservées dans les notes 🕳️, pas effacées. Le document publie désormais une
> **plage** pour le 2,07×, et **étiquette** la comptabilité de chaque colonne
> `b/poids` — voir la note de provenance en fin de fichier.

## La question

Le noyau fusé doit décoder chaque bloc de 24 poids dans le temps où le GPU
attend déjà la mémoire. Le décodeur d'archive (`Indexer::decode`) coûte
**869 ns/bloc** (machine au repos), soit **207×** le plancher d'un codeword
Golay reconstruit par XOR (4,2 ns). Que faire ?

## Ce qui a été mesuré

| banc | résultat |
|---|---|
| `decprofile` | le rang de permutation domine (~75-82 % après correction des biais du banc ; ~12-18 % d'allocations `Vec` cachées) |
| `classprofile` | blocs réels : 3-5 magnitudes (4 typique), **jamais 2** ; 50/50 pair/impair ; 57 % coquille 13 |
| `arrbits` (par coset) | arrangement : masques imbriqués **+0,342 b/poids** vs rang optimal ; positionnel +0,598 |
| `decfast` | récurrence `M' = M·c_j/n` en u64 : unrank **4,7×** plus rapide, bit-identique |
| `decfull` | **décodeur v1 complet : 243 ns/bloc, 3,3×**, bit-identique sur 200 766 points (bornes de classes incluses) |
| `llvq-metal/hello` | M3 Max : SIMD 32, 1024 threads/groupe, **2,52 T op/s** (chaîne dépendante — une latence, pas un débit crête) |
| `llvq-metal/decode` | **GPU, 16,7 M blocs** : sol 0,08 ns/bloc · **masques 0,11 ns/bloc (1,43× le sol)** · rang 8,27 ns/bloc (106×) |

## Les impasses, et pourquoi

- **Positionnel naïf** : +0,598 b/poids, 2× le coût des masques. Mort.
- **Rangs groupés** : transmettre la répartition des types entre groupes coûte
  plus que le rang économisé (+0,45 à +0,77). Mort.
- **« Arrangement = codeword Golay »** : enterré au départ pour une mauvaise
  raison (l'audit a montré que la structure Golay est au niveau *catégorie*
  dans 100 % des blocs, pas au niveau bloc) — mais le format v1 l'exploite
  déjà : le coset pair range séparément support et hors-support. Le gain
  résiduel est déjà dans les +0,383 du coset pair.
- **Décodage GPU coopératif des masques (ballot + prefix-sum)** : erreur de
  comptage de ma part, relevée par l'audit — une instruction simdgroup occupe
  32 lanes, donc ~35 instructions coopératives pour UN bloc = ~1120 lane-ops.
  Mort sous cette forme.

## Le piège u64 (audit)

Les *valeurs* des multinomials tiennent en u64 (< 2⁴⁸, car elles divisent le
budget d'index), mais **pas leur calcul par factorielles** : 21!..24!
débordent u64. Un décodeur u64 doit soit lire le multinomial initial dans la
table de classes, soit le maintenir par la récurrence — jamais recalculer
n!/Πc!. `decfull` fait les deux correctement.

## L'architecture qui en sort

Deux formats, un transcodage payé une fois :

| étage | format | coût | statut |
|---|---|---|---|
| disque | rang (v1) — **2,1595 b/poids**, le chiffre publié | optimal en bits | inchangé |
| chargement | transcodage rang → layout runtime | **~3,1 s** pour un 4B (12 cœurs, `decfull`) | mesuré |
| RAM | masques imbriqués, ~2,6-3,0 b/poids | +16-40 % de trafic vs 2,16 | conçu, non implémenté |
| noyau | **un bloc par lane**, décodage sériel des masques | **0,11 ns/bloc mesuré** → 16,8 ms/token sur un 4B | ✅ mesuré, viable |

Le surcoût des masques n'existe qu'en RAM : le fichier ne bouge pas.

## Les plafonds honnêtes (corrigés par l'audit)

Mon estimation initiale (« ~340-400 tok/s ») oubliait que le **lm_head lié
(778 Mo f16) est lu en entier à chaque token** pour les logits :

| | trafic/token | plafond à 400 Go/s |
|---|---|---|
| FP16 | 8,04 Go | ~50 tok/s |
| linéaires quantifiés seuls (2,6-3,0 b/poids) | 1,18-1,36 Go | 294-339 tok/s |
| **modèle réel, lm_head f16** | **1,96-2,14 Go** | **~187-204 tok/s** |

Soit **~3,8× le FP16** — pas ×7. Le levier suivant est identifié : quantifier
aussi le lm_head (il pèse 36-40 % du trafic une fois les linéaires à ~3 bits).

Bascule mémoire→calcul : ~185-200 opérations/bloc dans le scénario réel. Le
décodage bloc-par-lane des masques (~100-150) passe dessous ; le décodage
direct du rang (~509, compté par `decfast`) ne donnerait que ~1,5× le FP16.

## Le verrou est levé (2026-07-31, soir)

`llvq-metal/decode` mesure, sur GPU, 16,7 M blocs, sortie vérifiée contre une
référence CPU écrite indépendamment :

| noyau | par bloc | débit | vs sol |
|---|---|---|---|
| sol (aucun décodage) | 0,08 ns | 1,29·10¹⁰ blocs/s | — |
| **masques imbriqués** | **0,11 ns** | 9,01·10⁹ blocs/s | **1,43×** |
| rang (récurrence u64) | 8,27 ns | 1,21·10⁸ blocs/s | 106× |

**Décoder par masques coûte 43 % de plus que ne rien décoder.** Sur un 4B :
16,8 ms/token de décodage, contre 1252 ms pour le rang. La séparation
archive/runtime n'est pas un raffinement, c'est la condition d'existence du
noyau — 75× d'écart entre les deux formats.

### Trois défauts de banc corrigés en route, chacun changeant le résultat

1. La v1 **stockait** les 24 poids décodés : le « sol » sortait à 195 Go/s
   d'écritures non coalescées, un banc mémoire déguisé en décodeur. Un noyau
   fusé n'écrit jamais les poids décodés — le banc accumule désormais un
   produit scalaire et écrit un float par bloc.
2. L'activation était relue en mémoire globale à chaque itération (24 lectures
   par thread), plafonnant le sol à 80 Go/s. Elle est chargée une fois par
   threadgroup.
3. À 2 M blocs, le surcoût de soumission (~0,18 ms) valait le travail mesuré.
   16,7 M blocs le ramènent à 12 %.

Sans ces corrections le verdict était « 25 tok/s, c'est mort ». La vraie
valeur est 5× meilleure.

### Ce que ce chiffre ne couvre PAS

- **Tous les blocs ont 4 magnitudes** dans le banc ; les vrais en ont 3, 4 ou 5
  → la divergence entre lanes n'est pas testée.
- **Codes synthétiques**, pas de lecture réelle du flux depuis la RAM.
- **Pas de matvec** : ni réduction inter-lanes, ni tuilage, ni écriture de la
  sortie.
- Les 2,52 T op/s viennent d'une chaîne dépendante (latence), donc tous les
  budgets « opérations par bloc » restent ~2× pessimistes.

## Le format runtime, figé sur mesure (2026-08-01)

`rtbits` a fait la comptabilité **complète** — assignation + signes + classe +
gain + adressage, rien d'exclu — des layouts candidats, exhaustivement sur les
**150 681 600 blocs** du 4B publié (6,5 s : tout est au niveau classe, une
recherche binaire par index suffit). La distribution réelle colle au proxy
gaussien : 3-5 niveaux de magnitude, 4 pour 65,9 % des blocs, 50/50 pair/impair.

| layout (masques imbriqués) | b/poids | Go/token | plafond |
|---|---|---|---|
| borne inf. (adressage gratuit) | 2,9243 | 2,11 | 190 |
| **groupé 32, stride octet + base u32** | **3,3548** | 2,30 | **174 tok/s** |
| **champ fixe 96 bits (3×u32 alignés)** | **4,0000** | 2,59 | **154 tok/s** |
| archive v1 (48 b/bloc, indécodable en fusé) | 2,0000 | 1,69 | (237) |

**Morts par la mesure** : le positionnel (+0,8 b/poids sur chaque variante —
⌈lg L⌉ bits × 24 slots contre des masques qui rétrécissent), l'offset u16 par
bloc (3,59 — dominé par le groupé-32 qui n'en met qu'un par groupe), le champ
fixe 128 en nibbles (5,33 — dominé par le fixe-96, possible parce que le
**pire cas exact sur toute la table cap-13 est 74 bits**, classe 238, coquille
12 ; 96 bits couvrent tout bloc possible avec 22 bits de marge, pour toujours).

**Le payload est figé**, commun aux deux finalistes, bit-packé LSB-first :

```
[classe : 9 bits][gain : g bits][signes : nz bits][masques imbriqués]
```

- classe : 0 = origine, sinon 1 + rang de la classe dans la disposition v1
  (coquilles croissantes, paires puis impaires) ; 384 valeurs ≤ 2⁹.
- signes : un bit par coordonnée **non nulle**, ordre des slots, 1 = négatif.
  Le décodeur maintient un compteur de non-zéros en balayant les slots — il
  n'y a donc pas de popcount à payer pour trouver son bit de signe.
- masques : niveaux dans l'ordre **canonique** (comptes décroissants, égalité
  → valeur décroissante), masque k sur les slots que les niveaux < k ont
  laissés libres ; L−1 masques, le dernier niveau est implicite. Largeurs et
  valeurs se lisent dans la table des classes (384 entrées, constante).

**Tranché sur GPU le 2026-08-01** (`decreal`, 16,7 M blocs réels du 4B
publié, préfixes contigus des 252 matrices, chaque sortie vérifiée contre le
décodeur CPU — lui-même épinglé bit pour bit sur `Indexer::decode`, 10
mutants tués) :

| noyau | par bloc | vs sol | b/poids | plafond trafic |
|---|---|---|---|---|
| sol (12 o lus, rien décodé) | 0,084 ns | — | — | — |
| Fixed96, loads alignés | 0,152 ns | 1,81× | 4,000 | 154 tok/s |
| **Grouped32, strides octet** | **0,158 ns** | 1,89× | **3,355** | **174 tok/s** |

**Grouped32 est le format du noyau.** Ses loads non alignés et l'indirection
de base coûtent 4 % de calcul ; sa compacité rend 19 % de trafic. Le trafic
est la ressource rare — c'est toute la thèse de la quantification. F96 reste
dans le transcodeur comme variante de référence (même payload, adressage
trivial), utile au débogage du noyau.

Au passage, la question laissée ouverte par le banc synthétique est close :
le format **réel** — 3-5 niveaux mélangés, table des 384 classes, curseur de
bits — coûte 0,152-0,158 ns/bloc là où le squelette uniforme à 4 niveaux
coûtait 0,11. +38 %, dans l'épaisseur du trait.

⚠️ Ces ns/bloc sont des débits de la forme « un bloc par lane, chaîne
sérielle » — la forme la plus pessimiste. Ils comparent les variantes entre
elles et bornent le surcoût du décodage sur le travail FMA irréductible
(le sol) ; ils ne prédisent pas le débit du matvec fusé, qui réorganisera
le parallélisme d'instructions. C'est l'étape suivante, et la seule
restante : le matvec sur une couche, contre le FP16 de la même machine.

## Le matvec fusé, et le layout qu'il a imposé (2026-08-01, soir)

`llvq-metal/matvec` : gate_proj 9728×2560 du 4B publié — poids, centroïdes,
échelles de ligne et queue réels, sorties vérifiées à ~10⁻⁸·Σ|w·x| près
contre une référence CPU f64, mesuré à 32 dispatches par command buffer
(une matvec ≈ 150 µs, l'ordre du surcoût de soumission — le piège documenté).

| noyau | µs | Go/s eff. | vs FP16 |
|---|---|---|---|
| FP16 (half4, simdgroup/ligne) | 134,6 | 370 | 1,00× |
| sol G32 (mêmes loads, zéro décodage) | 49,2 | 224 | **2,73×** |
| LLVQ fusé, masques imbriqués (G32) | 206,5 | 53 | 0,65× |
| **LLVQ fusé, Flat32** | **155,0** | 106 | **0,87×** |

**Le baseline est honnête** (370 Go/s ≈ 93 % du pic machine) et **la forme
est bonne** (le sol bat le FP16 de 2,7×). Tout le déficit est l'ALU du
décodage — et il a fallu réviser la décision d'adressage prise le même jour :
les masques imbriqués, optimaux en bits, exigent des **popcounts de préfixes
en espace de rangs par slot**, un mur que trois réécritures du noyau n'ont
pas contourné (sans-branches 245 µs, niveaux packés deux passes 331,
demi-masques 189 — toutes battues par le naïf à 206).

**`Layout::Flat32`** attaque le mur côté format, ce que seul le transcodeur
paie : masques en **espace de slots** (un mot de 24 bits par niveau, niveau 0
implicite par complément), signes réordonnés **niveau-majeur**. Le noyau
itère chaque niveau par `ctz`, consomme ses signes séquentiellement, et ne
touche **jamais** un slot zéro. Coût : 4,54 b/poids sur cette couche (16,5 Mo
contre 11,0 en G32 imbriqué, contre 49,8 en FP16), pire cas 130 bits —
grouped-only. Verrouillé par les mêmes round-trips bit-exacts que les deux
autres layouts, la table GPU épinglée champ par champ, 8 mutants tués.

**Où on en est, dit sans fard** : le noyau fusé multi-coquilles existe,
il est juste, et il fait **0,87× le FP16** — pas encore le gain. Le budget
est précis : 85 µs d'ALU au-dessus du sol pour égaler le FP16, on en
consomme 106. Les 1,36-1,48× du papier (Table 7) sont une **coquille unique,
M = 3, sur GPU datacenter** — pas comparables. Leviers identifiés, par ordre :

1. **Trier les blocs par classe dans chaque groupe de 32** (+5 bits/bloc pour
   la position d'origine) : les lanes d'un même simdgroup décoderaient la
   même classe — plus de divergence dans les boucles de niveaux, tables
   uniformes. C'est le levier le plus prometteur et il est côté transcodeur.
2. Deux blocs en vol par lane (pipeline logiciel explicite).
3. Occupancy/taille de threadgroup (non exploré : 256 partout).

Et le rappel d'échelle : même à 0,87×, le modèle 2 bits tient là où le FP16
ne **rentre pas** — l'argument du projet reste la classe de modèle chargeable,
le débit est le second combat.

## La brèche : Slot32, 2,2× le FP16 (2026-08-01, nuit)

Un audit adversarial de 14 agents (4 lentilles × réfutation) a rendu trois
verdicts sur le banc du matvec, tous vérifiés puis intégrés :

**1. Le quatrième piège de mesure du projet.** Les buffers LLVQ (11-17 Mo)
tenaient dans le **SLC de 48 Mo** et étaient rejoués 576 fois — mesurés
cache-résidents, là où le FP16 (49,8 Mo > SLC) streamait la DRAM, c'est-à-dire
le régime que l'inférence réelle (1,8 Go/token) impose à tout le monde. Tous
les chiffres LLVQ antérieurs étaient des plafonds optimistes ; l'écart serré
« parité » n'était pas tranchable dans ce protocole. Corrigé : **4 copies de
chaque flux de poids en rotation** sur les 32 dispatches — l'empreinte
cumulée déborde le SLC pour tous. (Au passage : ce qui sérialise les 32
dispatches est le hazard WAW sur le buffer de sortie partagé, pas la
frontière d'encodeur — le harnais le documentait faux, corrigé.)

**2. Le tri par classe est un levier nul — en principe.** Les 32 lanes d'un
simdgroup exécutent le même ensemble de blocs quel que soit leur ordre
interne : le coût lockstep est invariant par permutation intra-groupe. Ce qui
avait fait gagner `Sorted32`, c'était l'extraction d'en-tête à offsets
directs, pas le tri — deux variables changées à la fois, la faute de méthode
que ce projet documente depuis les A/B de calibration. `Sorted32` reste
comme pièce de mesure.

**3. La brèche structurelle : `Layout::Slot32`.** Signes en **masque de
slots de 24 bits** → le payload entier vit à offsets fixes :

```
[classe 9][gain 1][smask 24][m₁..m₄ @ 24 bits]
```

La largeur ne dépend plus que de L. Le décodeur n'a plus besoin de nz, ni du
niveau zéro, ni des bases de signes, ni d'aucune boucle par niveau : 24 tours
fixes, niveau par 4 tests de masques (les absents sont zéro, le niveau 0 est
le défaut), signe par un bit, **zéro divergence, zéro état sériel**, quatre
chaînes indépendantes. Verrouillé comme les autres : round-trips bit-exacts
sur les 5 layouts, canonicité des octets testée (les bits de signe des slots
zéro sont nuls — un mutant l'a exigé), largeur unifiée table↔assert (un
autre mutant), 2 mutants équivalents identifiés comme tels.

**Résultat** — gate_proj 9728×2560 réel, protocole froid, best-of-15, stable
sur 4 runs :

| noyau | µs | Go/s eff. | vs FP16 | b/poids |
|---|---|---|---|---|
| FP16 (half4) | 139,8-141,8 | ~355 | 1,00× | 16 |
| sol (zéro décodage) | 51-53 | — | 2,7× | — |
| **LLVQ fusé Slot32** | **61,9-64,2** | ~275 | **2,20-2,26×** | 5,375 |
| LLVQ fusé Sorted32 | 135 | 124 | 1,04× | 4,75 |
| LLVQ fusé Flat32 | 156 | 105 | 0,90× | 4,54 |
| LLVQ fusé nested G32 | 208 | 53 | 0,68× | 3,35 |

🏷️ **Comptabilité de cette table** : payload + bases, sur **une seule couche**,
protocole froid à 4 copies rotatives. Ce n'est **pas** celle du modèle entier
plus bas, qui ajoute la queue f32 et les échelles de ligne f32 — les deux
colonnes `b/poids` ne sont pas commensurables, et c'est exactement ce que la
correction du 2026-08-05 est venue démêler.

Contrôle de non-régression rejoué le 2026-08-05, même binaire, même protocole :
**0,69×** (G32), **0,90×** (Flat32), **2,20×** (Slot32), **5,375 b/poids** sur
cette couche.

⚠️ Ne pas écrire « rien n'a bougé » : le contrôle rend **0,69×** là où la table
écrit 0,68×, et **2,20×** là où elle écrit une fourchette 2,20-2,26×. Le
dernier chiffre bouge, comme attendu d'un banc dont la dispersion est
documentée plus bas. Ce que le contrôle établit est plus modeste et suffit :
**aucun des trois rapports ne change d'ordre ni de conclusion**, et le
5,375 b/poids est reproduit à l'identique.

Le décodage Slot32 coûte **12 µs au-dessus du sol** — il se glisse dans les
bulles de latence mémoire, ce qu'un noyau fusé doit faire.

> 🕳️ **L'échelle bits↔vitesse publiée ici mélangeait trois protocoles et deux
> comptabilités. Corrigée le 2026-08-05 (lot K-1(a)).** Elle s'écrivait
> « 2,16 b/poids (archive, indécodable) → 3,35 (nested, 0,68×) → 4,54
> (Flat32, 0,90×) → **5,375 (Slot32, 2,21×)** », et se présentait comme
> « mesurée de bout en bout ». Elle ne l'était pas : les b/poids venaient de
> `rtbits`/`arrbits`, dont la comptabilité s'arrête au payload et aux bases ;
> les vitesses venaient de `matvec`, sur **une seule** couche (gate_proj) ; et
> le 2,16 de tête est le **fichier scellé**, qui n'est pas un layout runtime et
> qu'aucun noyau ne lit. Trois objets alignés comme s'ils étaient
> commensurables.
>
> Refaite dans **un seul run**, sur le **modèle entier**, avec la **même**
> comptabilité d'octets pour les quatre bras (`bin/thesis`, sept bras,
> 2026-08-05) :
>
> **3,498 b/poids → 0,69× [0,68–0,69]** (Grouped32) ·
> **5,256 → 0,91× [0,91–0,91]** (Flat32) ·
> **5,510 → 2,03× [2,03–2,10]** (Slot32), contre **16,000 → 1,00×** en FP16.
> Chaque rapport est la **médiane du rapport formé round par round**, avec sa
> plage sur les 5 rounds gardés — pas un quotient de deux minima. Journal :
> [`docs/mesures/k1-metal-2026-08-05.txt`](mesures/k1-metal-2026-08-05.txt).
>
> La table complète, avec sa comptabilité étiquetée, est en section
> « Le prix en RAM, et que c'est un cadran ». Ce que la correction change au
> fond : Flat32 ne coûte pas 4,54 b/poids mais **5,256**, à comparer aux
> **5,510** de Slot32. Les deux layouts sont bien plus proches en bits que la
> vieille échelle ne le laissait croire, et restent à **0,91× contre 2,09×**
> en vitesse — Flat32 n'est donc pas le compromis qu'il paraissait : il coûte
> presque autant de RAM que Slot32 sans en avoir la vitesse.
>
> ⚠️ **En revanche, la projection qui suivait — « ~3,2 Go/token → plafond
> ~124 tok/s » — n'est PAS caduque, et une première rédaction de cette note
> l'avait déclarée telle à tort.** `rtbits` l'imprime encore le 2026-08-05
> (« 5.376 b/p — 3.22 Go — 124 tok/s »,
> `docs/mesures/k1c-rtbits-2026-08-05.txt`). C'est un **plafond de bande
> passante**, pas un débit : il dit ce que le trafic d'octets autorise au
> mieux. `bin/thesis`, lui, **mesure** le pas de décodage. Deux quantités
> différentes qui ne se remplacent pas — et ne pas les confondre est le même
> réflexe que ne pas mélanger deux comptabilités.

### Reprendre des bits sans perdre la vitesse : le plafond L ≤ 4, compté

Mesuré par `rtbits` sur l'artefact réel `~/llvq-q4b.llvq` (981 Mo, projections
seules du 4B publié) — **4 708 800 groupes de 32 blocs, soit 150 681 600
blocs**. Comptabilité : payload + une base u32 par groupe, stride arrondi à
l'octet, c'est-à-dire ce qu'une lane lit réellement, rapporté aux poids
quantifiés.

> 🕳️ **Corrigé le 2026-08-05, deuxième passe.** La première rédaction de ce
> paragraphe écrivait « 113 011 200 blocs ». C'est faux de construction :
> 4 708 800 groupes de 32 blocs font 150 681 600 blocs, et ce document écrit
> déjà ce chiffre plus haut (section `rtbits`) — le fichier se contredisait
> donc lui-même à deux sections d'écart. Le 113 011 200 est
> `4 708 800 × 24`, c'est-à-dire un nombre de groupes multiplié par la taille
> d'un **bloc** au lieu de celle d'un **groupe** : deux unités confondues.
> Valeur mesurée : `docs/mesures/k1c-rtbits-2026-08-05.txt`, ligne
> « 150681600 blocs ».

| max de niveaux dans le groupe | groupes | part | stride |
|---|---|---|---|
| L = 3 | **1** | — | — |
| L = 4 | 1 566 710 | 33,272 % | 14 o |
| L = 5 | 3 142 089 | **66,728 %** | 17 o |

Moyenne des max par groupe : **4,667 niveaux** (moyenne par bloc : 3,726).
Le « ~2/3 des groupes à 17 octets » annoncé de mémoire est donc juste, et il
vaut **66,728 %**.

> ⚠️ **Coïncidence numérique, à ne surtout pas lire comme une égalité.** Les
> 4,667 ci-dessus sont des **niveaux** ; le 4,667 qui apparaît plus bas dans
> la note 🕳️ (et dans la table `lcap` de `docs/archive/face-au-4-bits.md`) est un
> **b/poids**. Deux quantités sans rapport qui tombent sur le même chiffre.

| | b/poids |
|---|---|
| `Slot32` aujourd'hui | **5,3756** |
| sous plafond L ≤ 4 | **≤ 4,7083** |
| gain | **0,667, soit 12,4 %** |

**Ce 4,7083 est un majorant inconditionnel, pas une simulation.** `L ≤ 4`
implique `width_slot ≤ 9 + 1 + 24×4 = 106 bits = 14 octets`, donc **tout**
groupe a un stride ≤ 14 o — aucune distribution ne peut faire mentir cette
ligne. Et le majorant est **atteint** dès qu'un groupe porte un bloc à
4 niveaux, parce qu'un bloc déjà à L ≤ 4 garde sa classe sous le plafond : son
mot de code est l'argmin sur la boule entière, il reste l'argmin sur un
sous-ensemble qui le contient encore. Compté sur l'artefact :
**4 708 799 groupes sur 4 708 800** portent un tel bloc. Ce n'est donc ni une
hypothèse d'indépendance ni une probabilité, c'est un compte.

Ce que le plafond ne dit **pas** : ce que deviennent les blocs L = 5. Les
re-quantifier change leur classe, et le coût en distorsion n'est pas mesuré
ici. Le gain en bits, lui, est acquis.

> 🕳️ **Le « ~4,4 b/poids » écrit ici jusqu'au 2026-08-04 était faux.** 4,4167
> est `106/24` : la largeur **brute** du bloc, ni arrondie à l'octet, ni
> chargée de la base u32 du groupe. Dans la comptabilité que le noyau paie
> réellement, la valeur est **4,7083** — 4,667 de payload à 14 octets, plus
> 0,042 d'adressage. Le paragraphe se contredisait d'ailleurs tout seul : il
> écrivait « 14 octets » deux lignes plus haut, et 14 octets font 4,667
> b/poids à eux seuls, donc le total ne pouvait pas être inférieur. Même
> motif que les autres prises du dossier — un chiffre annoncé dans une
> comptabilité et lu dans une autre.

## La thèse, sur le modèle entier (2026-08-01, complétée le 2026-08-05)

`bin/thesis` mesure la revendication elle-même : **un token de projections**,
les 252 matrices du 4B publié, un command buffer par format.

La forme est honnête par construction, et c'est le point : une passe touche
2,50 Go (LLVQ) ou 7,27 Go (FP16) de poids **distincts** — rien n'est relu, donc
aucune matrice ne peut se cacher dans le SLC de 48 Mo. Là où `bin/matvec` doit
forcer le régime froid avec 4 copies en rotation, ici le travail réel le donne.
Une seule soumission pour 252 dispatches, sérialisés par le hazard WAW sur la
sortie partagée — le même mécanisme que la dépendance entre couches.

| | ms/token | Go lus | Go/s | vs FP16 |
|---|---|---|---|---|
| FP16 (half4) | 21,691 | 7,27 | 335 | 1,00× |
| **LLVQ fusé (Slot32)** | **10,460** | 2,50 | 239 | **2,07×** |

Avec le `lm_head` lié (389 M poids f16, non quantifié, identique aux deux
côtés, 2,32 ms) : **41,6 → 78,2 tok/s**, ×1,88. C'est lui qui plafonne le
rapport de bout en bout, et c'est le levier suivant identifié depuis juillet.

**1 105 920 lignes vérifiées** contre une référence CPU f64 avant toute mesure
— pire erreur LLVQ 3,4·10⁻⁸·Σ|w·x|, FP16 2,8·10⁻⁸.

⚠️ **Ce 2,07× est le haut d'une plage, pas une valeur ponctuelle** — voir
juste en dessous. Il a été publié comme un point, et l'affirmation
« reproductible : 2,07× et 2,08× sur deux passes » qui figurait ici décrivait
en réalité deux runs déjà réchauffés.

🔎 **Le fusé n'est pas encore limité par la mémoire, et c'est une marge.** Le
FP16 tire 336 Go/s (93 % du pic) : il est au mur de la bande passante, il n'y
a rien à en tirer de plus. Le LLVQ n'en tire que **240** — donc il reste
borné par le calcul du décodage. Au même débit mémoire que le FP16, ses
2,50 Go passeraient en **7,5 ms**, soit **2,9×** au lieu de 2,07. C'est le
plafond de la forme actuelle, et il se prend sans changer un bit du format.

### La dispersion inter-processus, et ce qu'elle interdit (2026-08-05)

Le banc à deux bras, **strictement non modifié**, lancé trois fois de suite :

| run | FP16 ms | LLVQ ms | rapport |
|---|---|---|---|
| 1 | 21,983 | 10,832 | **2,029** |
| 2 | 21,783 | 10,627 | **2,050** |
| 3 | 21,680 | 10,421 | **2,080** |

Les erreurs (3,4·10⁻⁸ LLVQ, 2,8·10⁻⁸ FP16) et les octets lus sont identiques
aux trois runs : l'arithmétique n'a pas bougé, seuls les temps bougent. Et ils
bougent **monotonement**, les deux bras accélérant ensemble — ce n'est pas du
bruit symétrique autour d'une valeur vraie, c'est un réchauffement qui vit
**entre** les processus, là où le warm-up interne (7 passes, 2 jetées, minimum
des 5) ne peut rien : cache de fichier sur les 981 Mo d'artefact, état
d'horloge du GPU, résidence des buffers.

**Donc le 2,07× publié est le haut de la plage [2,029 ; 2,080]**, reproduit au
**troisième** run consécutif, pas son centre.

**Ce que ça interdit est le vrai résultat de la mesure** : un effet de
quelques pour cent ne peut pas être tranché en comparant deux invocations
distinctes du binaire. Les bras doivent être entrelacés **dans un même
processus**, tous dispatchés à chaque round dans le même ordre, et leur
dispersion imprimée. C'est le protocole du run à sept bras plus bas, et c'est
ce qui rend lisible son verdict sur le padding, où l'écart mesuré est de
**0,4 %** (10,081 contre 10,126 — grandeur dérivée de la table du run à sept
bras, pas une ligne lue).

### Le trou de couverture que ça a fermé

`bin/matvec` stageait toute l'activation en mémoire threadgroup : 10 Ko à
d_in = 2560, mais **38 Ko à d_in = 9728 contre la limite Metal de 32 Ko**. Les
36 `down_proj` du modèle n'auraient pas pu tourner — le 2,2× était mesuré sur
une forme qui passe. Les deux noyaux de `thesis` tuilent l'activation par
128 blocs (3072 colonnes, 12 Ko), donc les six formes du modèle empruntent le
même code.

### Le prix en RAM, et que c'est un cadran

Sur le modèle entier, `Slot32` coûte **5,510 b/poids** dans la comptabilité de
`bin/thesis` et **5,3756** dans celle de `rtbits`. Le fichier, lui, ne bouge
pas — 2,1696 b/poids. D'un même `.llvq` on charge le format qu'on veut.

> 🕳️ **L'écart 5,375 / 5,51 était imputé ici aux « autres formes de matrices,
> autres distributions de classes et autres arrondis de stride ». C'est faux —
> corrigé le 2026-08-05.** L'écart est une différence de **comptabilité**, pas
> de matrice : `rtbits` compte le payload plus une base u32 par groupe ;
> `bin/thesis` y ajoute la queue f32 et les échelles de ligne f32. Deux
> numérateurs pour le même objet, pas deux mesures contradictoires.
>
> Preuve directe, sans recalcul : `rtbits` sur le modèle **entier** rend
> **5,3756**, et `bin/matvec` sur **gate_proj seule** rend **5,375**. Si
> c'était un effet de forme de matrice, ces deux-là ne coïncideraient pas.

**Les quatre layouts, un seul run, une seule comptabilité** (`bin/thesis`,
sept bras, 2026-08-05, journal
[`docs/mesures/k1-metal-2026-08-05.txt`](mesures/k1-metal-2026-08-05.txt) ;
252 projections du 4B publié, un command buffer par
bras, mémoire froide par construction ; 7 rounds dont 2 jetés, **tous** les
bras dispatchés à chaque round dans le même ordre ; colonne ms = min des
5 rounds gardés ; comptabilité d'octets **identique aux quatre bras** =
payload + bases + queue f32 + échelles de ligne f32 ; 1 105 920 lignes
vérifiées contre une référence CPU f64, seuil 1e-5) :

| en RAM | b/poids | projections | ms/token (min) | Go/s | vs FP16 (méd) [plage] |
|---|---|---|---|---|---|
| FP16 (half4, scalaire) | 16,000 | 7,27 Go | 21,728 | 334 | 1,00× [1,00–1,00] |
| `Grouped32` | 3,498 | 1,59 Go | 31,634 | 50 | 0,69× [0,68–0,69] |
| `Flat32` | 5,256 | 2,39 Go | 23,807 | 99 | 0,91× [0,91–0,91] |
| **`Slot32` (scalaire@24)** | **5,510** | **2,50 Go** | **10,496** | 241 | **2,03× [2,03–2,10]** |

⚠️ **La colonne « vs FP16 » n'est pas le quotient des deux colonnes de gauche.**
C'est la **médiane du rapport formé round par round**, avec sa plage sur les
5 rounds gardés. Diviser un minimum par un minimum mêlerait deux rounds qui
n'ont jamais coexisté : un lecteur qui pose 21,728 / 10,496 trouve 2,09 ici,
mais 21,728 / 10,126 trouve 2,19 pour la variante `float4` plus bas, là où le
rapport round par round donne 2,15. C'est la même précaution que la section
sur la dispersion : ce sont les **rapports**, pas les millisecondes, qui se
comparent.

Les millisecondes **dérivent d'un run à l'autre** — c'est le fait même que ce
lot a établi. Les `b/poids` et les octets, eux, sont **exacts** et se
reproduisent au chiffre.

Le bras `Slot32` est le noyau de la table de tête de cette section, remesuré :
**2,03× [2,03–2,10]** ici contre **2,07×** là. Les deux plages se recouvrent —
[2,05–2,11] contient le 2,07× de la table de tête —, et le 2,09× déborde par
le haut la plage inter-processus [2,029 ; 2,080] documentée ci-dessus. Les
deux rapports ne sont d'ailleurs pas formés de la même façon : ici round par
round dans un même processus, là par quotient de deux minima. Ne pas les
soustraire.

Même au plus large, le modèle chargé fait ~3,3 Go contre 8,045 en FP16.

#### Expérience : le conflit de bancs (K-1(b))

`docs/archive/portage-noyau-cuda.md` §3.2 prédisait, en transposant l'arithmétique
NVIDIA (32 bancs de 4 octets), un conflit à 8 voies sur chacune des 24 lectures
de tuile, et prescrivait un **pas de 28 flottants** avec chargements `float4`
pour le supprimer. Trois bras de plus, dans le **même** run et la **même**
comptabilité que la table ci-dessus, l'ont testé. Le « vs FP16 » reste rapporté
au FP16 scalaire.

| bras (expérience) | b/poids | ms/token (min) | Go/s | vs FP16 (méd) [plage] |
|---|---|---|---|---|
| FP16 (half4, **float4**) | 16,000 | 20,612 | 351 | 1,05× [1,04–1,07] |
| `Slot32` (**float4**@24) | 5,510 | 10,126 | 252 | **2,14× [2,10–2,16]** |
| `Slot32` (float4@**28**) | 5,510 | 10,081 | 248 | 2,13× [2,07–2,17] |

Même convention que la table précédente : le « vs FP16 » est la **médiane du
rapport formé round par round**, avec sa plage sur les 5 rounds gardés, et non
le quotient des colonnes `ms` — 21,728 / 10,126 donnerait 2,19, borne haute de
la plage, pas sa médiane.

**Le modèle de bancs NVIDIA n'est pas valide sur Apple.** Les trois points
ci-dessous sont des grandeurs **dérivées** de la table — licites parce que
leurs deux termes viennent du même run et de la même comptabilité, mais
dérivées, pas lues :

1. Passer du scalaire au `float4` gagne **3,5 %** sur LLVQ (10,496 → 10,126)
   **et 5,1 %** sur FP16 (21,728 → 20,612). Les deux bras, presque autant.
2. Le **padding à 28 ne gagne rien** : 10,081 contre 10,126, soit **0,4 %**
   *plus lent*, et sa plage de rapport [2,06–2,17] recouvre par le haut celle
   du dense [2,12–2,19]. Rien ne le distingue — et un tel écart, la section
   sur la dispersion le dit, n'est pas tranchable.
3. À bras comparables, le rapport ne bouge pas : `float4` contre `float4`
   donne **2,04×** (2,15 / 1,05), contre **2,09×** en scalaire contre
   scalaire. Les deux sont dans la dispersion l'un de l'autre.

Ce qui paie n'est donc pas la suppression d'un conflit de bancs, c'est la
**largeur de chargement** — un load 128 bits au lieu de quatre loads 32 — et
elle paie des deux côtés, donc elle ne change pas le rapport.

⚠️ **Le confondant, déclaré :** les deux variantes `float4` de `Slot32` sont
identiques **au bit près** au noyau scalaire sur les 1 105 920 lignes (une
assertion du code l'exige). La variante `float4` du bras FP16 ne l'est pas —
3,1·10⁻⁸ d'écart — parce que sa somme est écrite en `+`/`*` et non en `fma`
explicites, donc le compilateur contracte comme il veut. Les 5,1 % gagnés
côté FP16 ne sont pas garantis à arithmétique constante.

### Ce que ce chiffre ne couvre pas

Attention, normes, activations et la rotation de `x` ne sont pas mesurées —
seulement la part que la quantification change.

> ✅ **Corrigé le 2026-08-08 : « le noyau n'est pas branché dans `bin/run` …
> c'est le dernier chantier » n'est plus vrai depuis le 2026-08-06.** Le noyau
> est branché **sur CUDA** : `fused_cuda` remplace les 252 `Linear` par deux
> lancements chacun (rotation puis matvec fusé), et son appelant est
> `bin/fusedrun`. Mesuré sur L40S, 128 tokens, sur les octets publiés :
> **48,7 tok/s dans 2,96 Go contre 43,6 dans 8,04**, mêmes tokens gloutons
> jusqu'à un tie-break au token 89
> ([`mesures/planes14-fusedrun-2026-08-06.txt`](mesures/planes14-fusedrun-2026-08-06.txt)).
> Avec l'embedding int8 au chargement : **88,4–88,5 tok/s dans 2,60 Go**.
> ⚠️ **Le ×2,03 de ce second chiffre n'est pas le noyau Leech** — ~25 ms/token
> viennent du remplacement de **notre** chemin `lm_head` dense, qui appelle
> `broadcast_matmul` et recopie 778 Mo par token. Les modèles de
> `candle_transformers` passent par `Linear` et ne paient pas cette copie
> ([candle#3871](https://github.com/huggingface/candle/issues/3871)) : la
> baseline handicapée est la nôtre. **Le rapport à tête
> identique est ×1,12**, et c'est celui qui mesure le noyau
> ([`mesures/phases-2026-08-07.txt`](mesures/phases-2026-08-07.txt)).
>
> Deux réserves qui, elles, tiennent. **Sur Metal, rien n'est branché** :
> `llvq-metal` reste un banc, `bin/run` décode en mémoire sur CPU comme sur
> Metal. Et la **rotation de `x`**, non mesurée ici, l'est désormais sur
> CUDA — `rot_apply`, vérifié contre une référence f64 sur huit formes, 8,05 µs
> à n = 2560 en isolation
> ([`mesures/rotation-cuda-2026-08-05.txt`](mesures/rotation-cuda-2026-08-05.txt))
> — et le bout-en-bout ci-dessus la paie déjà.

⚠️ Correction au passage : l'estimation « ~2,6-3,0 b/poids » de la table
d'architecture ci-dessus ne comptait ni la classe, ni l'adressage. Le vrai
plancher adressable est 3,35 **dans la comptabilité `rtbits`** (payload +
bases ; 3,498 dans celle de `bin/thesis`, qui ajoute queue et échelles de
ligne) ; les plafonds deviennent 174/154 tok/s, toujours ~3,1-3,5× le FP16, et
le lm_head pèse toujours le tiers du trafic.

## Le plancher de NOTRE géométrie — ce qu'aucun de NOS formats ne touche (2026-08-16, réinterprété le 2026-08-21)

> 🚨 **Le titre de cette section a dit « ce qu'aucun format ne touche » pendant
> cinq jours, et il portait deux mots de trop.** Le chiffre est intact : une
> passe de projections qui ne lit aucun poids coûte **2,305 ms**, rejouée à
> **2,306** dans le banc à dix bras du 2026-08-21, soit 45,2 % du bras servi.
> Ce qui est faux est ce que le document en tirait — « **plafond absolu de tout
> travail de format : 4,77× FP16** ». **Un noyau réel passe dessous** : QTIP
> rend **2,246 ms** sur les mêmes 252 projections, dans le même processus, en
> lisant 0,91 Go
> ([`mesures/f2-p3-qtip-banc-2026-08-21.txt`](mesures/f2-p3-qtip-banc-2026-08-21.txt)).
>
> **Le mécanisme, et il n'est pas anecdotique** : `tv_nullk` a été écrit pour
> partager **la grille des bras servis** — c'était sa qualité, c'est ce qui rend
> la soustraction licite — donc il hérite de **notre** géométrie : **un warp par
> ligne de sortie, 252 lancements**. QTIP est lancé dans **la sienne**
> (`<<<128, 1024, 64 Kio>>>`, 252 lancements aussi). Ce que `nullk` borne est
> notre **structure de noyau**, pas la machine. Un noyau de forme différente la
> traverse — et c'est exactement ce que la mesure montre.
>
> **Conséquence de rédaction, à appliquer partout dans ce dépôt** : on écrit
> « **plancher de notre géométrie de lancement** », jamais « plancher machine »
> ni « plafond absolu ». Les « 39 % de latence/occupation » de l'attribution du
> 2026-08-05 tombent sous la même réserve : ce sont des propriétés de notre
> géométrie, pas des constantes de la carte.

Tout ce document mesure le **décodage** : ce qu'il coûte, ce qu'il économise, à
quel prix en bits. Il lui manquait le **dénominateur** — ce que coûte le noyau
quand il ne décode rien du tout. Ce reste n'avait jamais été qu'une
**soustraction** (l'attribution du 2026-08-05, 39 % de « latence/occupation »).
C'est un **chiffre** depuis le 2026-08-16.

`tv_nullk` garde **la grille, le tuilage, les deux barrières, le staging de
l'activation, `warp_sum`, l'épilogue de queue et l'écriture de `y`** — et
retire la lecture et le décodage du bloc. 31 registres, 0 octet local. Même
banc, même run, mêmes rounds que les bras servis
([`mesures/nullk-plancher-2026-08-16.txt`](mesures/nullk-plancher-2026-08-16.txt),
job `6a81b2b71f5885ae605bdcc9`, L40S, **0,77 $** ; 252 projections d'un token,
7 rounds dont 2 jetés, rapports formés **round par round**).

| bras | ms (médiane) | |
|---|---|---|
| **plancher (`nullk`) — aucun poids lu** | **2,305** | **45,2 % du bras servi** |
| `Planes14` — le layout servi | 5,102 | |
| FP16 | 10,996 | |

**Ce que ça borne, et ce que le format achète vraiment** — toutes grandeurs
du même run :

| | |
|---|---|
| **borne de tout travail de format *dans notre géométrie*** | **4,77× FP16** [4,74–4,77] — ⚠️ **pas** un plafond absolu, cf. le 🚨 en tête de section |
| où `Planes14` en est | **2,16×** [2,15–2,16] |
| ce que le format achète **net du plancher** | **3,11×** — 8,691 ms de trafic contre 2,797 |
| coût du décodage de `Planes14` | **~7 %** du temps de trafic (779 Go/s nets contre 836) |

🕳️ **La première ligne s'intitulait « plafond absolu de tout travail de
format » ; le nombre est le même, l'adjectif est retiré.** Il ne borne que les
noyaux qui partagent la grille de `nullk` — c'est-à-dire les nôtres. Le
2026-08-21 mesure un noyau qui ne la partage pas et qui la franchit : QTIP rend
**4,89× [4,89–4,90]**, contre un `nullk` à 4,77× dans le **même** processus.
Toute phrase de ce document qui commence par « aucun format ne peut dépasser »
doit se relire « aucun de nos formats, dans cette géométrie, ne peut dépasser ».

**Dans notre géométrie, le format se dispute au plus 55 % du temps, et
`Planes14` en capture déjà l'essentiel.** C'est le renversement que ce document
doit porter : pendant que quatre campagnes cherchaient à descendre sous
`Planes14` **en bits**, le poste majoritaire n'avait jamais été attaqué. Le
chiffre qui l'aurait dit coûtait 0,77 $ et un noyau de trente lignes.
⚠️ **Et « majoritaire » ne veut pas dire « irréductible »** : ce poste-là est
celui d'une géométrie, donc il se déplace en changeant de géométrie — ce que la
fusion de `D1` fait pour 108 lancements, et ce que QTIP fait en entier.

Les débits **nets du plancher**, sur les bras qui partagent sa grille — la
seule lecture qui isole le décodage :

| bras | total ms | net ms | Go/s nets |
|---|---|---|---|
| FP16 | 10,996 | 8,691 | 836 |
| **`Planes14`** | **5,102** | **2,797** | **779** |
| `Slot32` | 5,824 | 3,519 | 710 |
| `Planes12x` | 5,498 | 3,193 | 617 |
| `Golay70` v1 | 8,223 | 5,918 | 275 |

**275 Go/s nets pour `Golay70` v1 : c'est là qu'un décodeur lourd se voit**, et
c'est la signature qu'E1v pousse à l'extrême — 25 Go/s, 0,25× FP16.

### Les quatre routes fermées sous `Planes14` — et la cinquième, restée ouverte

| route | ce qu'elle pesait | vitesse **mesurée** | verdict |
|---|---|---|---|
| **E3** (décoder l'index du fichier) | 3,0444 b/poids noyau contre un critère de 2,60 | jamais écrite | ❌ enterrée **sur papier**, 0 $ (2026-08-12) |
| **`Golay70` v2** | 3,589 b/poids noyau | **1,77× [1,76–1,78]**, 263 Go/s | ❌ seuil pré-enregistré de **2,0×** (2026-08-11) |
| **`e1c14`** aligné warp, **formes du 4B** | 5,2354 b/poids noyau contre 4,8040 — **+9,0 %** | jamais mesurée | ❌ **plus gros** que ce qu'il remplace **au 4B** (2026-08-16) ; ⚠️ **NE TRANSFÈRE PAS** — sur les formes du 14B il rend **4,6410 contre 4,7063, −1,4 %** (2026-08-17) |
| **E1v** | 2,3877 b/poids noyau (2,3983 aligné) | **0,25× FP16 [0,25–0,25]**, 25 Go/s | ❌ plancher d'X3 de **1,60×**, manqué d'un facteur **6,4** (2026-08-16) |
| `e1c12` aligné warp | **4,2880** contre 4,3424 pour `Planes12x` — **−1,3 %** *(formes du 4B ; **−10,4 %** sur celles du 14B, 3,8021 contre 4,2420)* | **jamais mesurée** | ❔ **ouvert**, et c'est désormais une question de **vitesse** |

**Aucune n'est bornée en octets ; toutes le sont en calcul.** E1v le montre au
plus net : il **tient sa promesse mémoire au bit près** — 1,09 Go lus contre
2,18 pour `Planes14`, la moitié, sur carte et sur le modèle publié — et il
**perd d'un facteur 8,7 sur le layout servi**, avec un noyau par ailleurs
exact (2,4e-8·Σ|w·x| sur 1 105 920 lignes), à 79 registres et **zéro spill**
([`mesures/e1v-cuda-2026-08-16.txt`](mesures/e1v-cuda-2026-08-16.txt),
0,85 $). Ce qui est mort est le **décodeur en ligne**, pas le format : celui-ci
reste disponible **hors boucle** (disque, transport).

⚠️ **Et `e1c12` n'hérite PAS du verdict d'E1v.** E1v meurt de son ALU — deux
marches binomiales, un mot de Golay, une réparation de parité, trois règles de
signe. `e1c12` décode le **même contenu que `Planes12x`**, c'est-à-dire des
sélections sur des plans de bits : sa question est un **motif de lecture**, pas
une charge d'ALU
([`mesures/e1c12-aligne-2026-08-16.txt`](mesures/e1c12-aligne-2026-08-16.txt),
0 $). Le pronostic ne se transporte pas.

### Quatre gardes, toutes dans le journal, aucune optionnelle

1. **AWQ n'est pas dans le tableau des nets, et son absence est la garde.** Son
   net donnerait **2 006 Go/s**, au-dessus de la HBM d'une L40S. Son noyau a sa
   **propre grille** (`awq_gemv.cu`), donc ce plancher-ci ne s'en soustrait
   pas. Un chiffre impossible qui dit exactement où la soustraction cesse
   d'être licite.
   🆕 **QTIP tombe sous la même garde, et de la façon la plus nette
   possible** (2026-08-21) : sa soustraction rendrait un net **négatif**
   — 2,246 ms de total contre 2,306 de plancher. Cette garde avait donc
   raison avant la lettre, et le journal du 08-16 l'écrivait déjà : la
   soustraction n'est licite qu'entre bras qui **partagent la grille**. Ce
   que personne n'avait vu, c'est que la conclusion symétrique — « donc
   `nullk` borne tout le monde » — ne suivait pas.
2. 🚨 **Ce 45,2 % n'est PAS les « 39 % » de l'attribution du 2026-08-05.**
   Celle-ci découpe **2,04 ms par token**, normes, attention et rotation
   comprises ; celui-ci mesure **252 projections d'un token**. Deux
   dénominateurs — les rapprocher demande de **refaire l'attribution**, pas de
   reporter un nombre. C'est la même faute que celle des trois comptabilités
   RAM ci-dessous, sur un autre axe.
3. **Le plancher n'est pas du gaspillage.** Staging, réduction, épilogue et
   écriture sont du travail qu'un noyau réel doit faire. C'est un **plancher**,
   pas une perte — mais c'est un **plafond** sur ce que le format peut gagner.
   ⚠️ **Amendé le 2026-08-21** : « du travail qu'un noyau réel doit faire »
   reste vrai *à grille fixée* ; ce n'est pas un invariant du problème. QTIP
   fait le même travail de projection pour **moins** que ce plancher, donc une
   part de ce que nous facturons ici est le prix de **notre** découpage, pas du
   travail lui-même.
4. ✅ **La garde n° 4 a été levée, et pas par où elle regardait (2026-08-24).**
   🕳️ Elle disait : « ⏳ **Rien sur `k > 1`.** La famille `k` de P4 §2.6 est
   écrite pour amortir exactement ce plancher, et **elle n'existe pas**. C'est
   le seul levier nommé du dépôt qui vise ces 45 %, et personne ne l'a
   chiffré. » Les deux premières phrases tiennent — la famille `k` n'est
   toujours pas écrite. La troisième était fausse **au moment où elle
   nommait un unique levier** : le poste se réduit aussi en **fusionnant les
   lancements**, ce qui ne change ni le format ni la forme du noyau, seulement
   leur nombre. `D1` l'a mesuré sur le **chemin servi**, pas au banc :
   **252 → 144 matvec par token, ×1,061 [1,050–1,069]** à `ROT_SHARE`
   constant, pour **+3 686 400 octets exactement** (+0,008117 b/poids), six
   critères pré-enregistrés verts — 128 tokens identiques entre bras fusé et
   non fusé, divergence au dense au même token 89, **même sha256 de source
   NVRTC sur les deux bras** (donc pas un artefact d'unité de traduction)
   ([`mesures/d1-fusion-servie-2026-08-24.txt`](mesures/d1-fusion-servie-2026-08-24.txt),
   job `6a8c6fbc…`, 0,24 $).
   **La décomposition, trois points tous mesurés sur cette carte** : **87,0**
   tok/s (`ROT_SHARE=0, FUSE=0`, la configuration servie publiée **à cette date**) → **94,9**
   (`ROT_SHARE=1, FUSE=0`, le hissage de rotation seul, ×1,091) → **100,6
   [99,9–100,7]** (`ROT_SHARE=1, FUSE=1`, plus la fusion, ×1,061).
   ⚠️ **Les deux marches sont deux mécanismes, et une seule des deux se
   publie** : le ×1,091 du hissage est une lecture **inter-jobs** — les 87,0
   viennent de B2, le 2026-08-18, sur une autre unité de traduction — donc il
   *se rapporte* et **ne se publie pas** comme une mesure de ce lot. Le ×1,061
   de la fusion est **intra-job**, à unité de traduction identique et
   `ROT_SHARE` constant ; `check_fuse` refuse d'ailleurs la paire
   `FUSE=1 + ROT_SHARE=0`, pour que le delta ne porte jamais deux mécanismes.
   🚨 **Et le 11,7 % du banc sur `Planes14` (5,096 → 4,504 ms) NE SE TRANSPORTE
   PAS** : il est mesuré en **f32**, sur `tv_planes_seg`, **hors modèle**, sur
   le **temps matvec seul**, quand ce lot mesure des tok/s bout-en-bout en f16.
   11,7 % du temps **matvec** et 6,1 % du temps **par token** sont deux
   quantités différentes, cohérentes entre elles — pas deux versions d'un même
   chiffre.
   ⚠️ **Ce que le lot ne mesure pas** : le 8B et le 14B n'ont **pas** été
   rejoués sous fusion. La table à trois tailles reste donc sur **une seule**
   configuration (`ROT_SHARE=0, FUSE=0`) aux trois tailles — propriété qu'elle
   utilise, et qu'un 4B fusé isolé casserait.
   🕳️ **LEVÉ le 2026-08-31 (vague 2)** : la fusion est rejouée et verte aux
   trois tailles (×1,055 au 8B, ×1,028 au 14B — 280→160 lancements, ses 40
   couches — bande [1,00 ; 1,12] tamponnée `e23e9895…`), et la **config servie
   v1 EST la fusée** : `planes14+q8+ROT_SHARE=1+FUSE=1`, **100,6 / 75,5 /
   46,8 tok/s dans 2,57 / 5,41 / 9,40 Go**. La propriété « une seule
   configuration partout » est préservée **par le gel**, plus par l'interdit
   ([`mesures/vague2-fusion-8b-14b-2026-08-31.txt`](mesures/vague2-fusion-8b-14b-2026-08-31.txt)).
   ⚠️ La série à **tête identique** (×1,11 → ×1,29 → ×1,41), la seule qui
   mesure le noyau, reste à `ROT_SHARE=0/FUSE=0` et n'est **pas** re-mesurée
   sous v1 — les deux formulations se donnent toujours ensemble.

🕳️ **Un défaut du journal, qu'il relève lui-même plutôt que de le laisser.** Sa
ligne V0 imprime « pires erreurs nullk 0.0e0·Σ|w·x| », ce qui se lit comme un
accord parfait avec la référence. **C'est faux** : ce bras n'a **aucun étalon**
et n'est pas comparé — ce qui a tourné est un contrôle d'observabilité (sortie
majoritairement non nulle). Corrigé dans le banc après le run ; qui lit ce
fichier doit lire ce `0.0e0` comme « non comparé », jamais comme « exact ».

## Le banc à dix bras — QTIP dans notre harnais, et le plancher qui n'en est pas un (2026-08-21)

Le lot F2 a porté le noyau de **QTIP** — quantification à treillis, 2 bits —
dans **notre** banc : même processus, mêmes 252 matrices du 4B publié, mêmes
formes, bras entrelacés, 7 rounds dont 2 jetés, rapports formés **round par
round**
([`mesures/f2-p3-qtip-banc-2026-08-21.txt`](mesures/f2-p3-qtip-banc-2026-08-21.txt),
job `6a881ea0…`, L40S, 0,89 $ ; pré-enregistrement
`proofs/preregistration-f2-qtip-2026-08-20.md`, tampon `.ots` posé **avant** le
premier job). C'est la première fois qu'un concurrent à 2 bits est chronométré
ici sans changer de harnais, et c'est ce qui rend le verdict lisible.

> ⚠️ **Dette, et elle porte sur tous les « tampon `.ots` posé d'avance » de ce
> document.** Les **16** ancrages `.ots` de `proofs/` (pour 22 documents de
> pré-enregistrement) n'ont **jamais été upgradés** : vérifié le 2026-08-25,
> tous portent **4 `PendingAttestation` et 0 `BitcoinBlockHeaderAttestation`**.
> Ce qui existe est donc une **soumission** aux calendriers, pas une preuve
> d'antériorité vérifiable par un tiers hors ligne. Ça ne retire rien aux
> verdicts — le pré-enregistrement est aussi commité, daté par git, et son
> sha256 est reproductible — mais on n'écrit pas « ancré dans Bitcoin » tant
> que l'upgrade n'a pas tourné.

🏷️ **Comptabilité de cette table** : `b/poids` **noyau** — payload + bases +
queue f32 + échelles de ligne f32 —, `Go lus` mesurés par le banc, `Go/s` sur
le temps **min**, `vs FP16` **médiane du rapport formé round par round** avec
sa plage sur les 5 rounds gardés. Ce n'est **pas** le quotient des colonnes.

| bras (phase 2/2, 10 entrelacés) | méd ms | Go lus | b/poids noyau | Go/s (min) | vs FP16 [plage] |
|---|---|---|---|---|---|
| plancher `nullk` — aucun poids lu | 2,306 | 0,07 | 0,159 | 31 | 4,77× [4,76–4,77] |
| FP16 (128 bits, témoin maison) | 10,994 | 7,27 | 16,000 | 661 | 1,00× |
| FP16 cuBLAS | 10,830 | 7,27 | 16,000 | 672 | 1,02× [1,02–1,02] |
| `Slot32` | 5,820 | 2,50 | 5,510 | 431 | 1,89× [1,89–1,89] |
| **`Planes14` — le layout servi** | **5,103** | **2,18** | **4,804** | **428** | **2,15× [2,15–2,16]** |
| `Planes12x` | 5,492 | 1,97 | 4,342 | 359 | 2,00× [2,00–2,00] |
| `Golay70` v2 | 6,182 | 1,63 | 3,589 | 264 | 1,78× [1,77–1,78] |
| `Golay70` v1 | 8,187 | 1,63 | 3,589 | 199 | 1,34× [1,34–1,34] |
| **AWQ w4g128** — 🏁 **CONCURRENT** | 3,252 | 1,90 | 4,179 | 584 | 3,38× [3,37–3,38] |
| **QTIP 2 bits** — 🏁 **CONCURRENT** | **2,246** | **0,91** | **2,000** | **405** | **4,89× [4,89–4,90]** |

🚨 **Sur les deux lignes 🏁, la grandeur comparable est les `Go/s`, PAS le
`×`.** Le banc l'imprime lui-même à chaque round. Un `×` vs FP16 récompense
d'abord le fait de lire moins d'octets : QTIP en lit **2,40× moins** que
`Planes14` *(calculé : 2,18 ÷ 0,91, deux colonnes du même run)*, donc son 4,89×
n'est pas « 2,3 fois meilleur noyau ». En Go/s
convertis, l'ordre est **AWQ 584 · `Planes14` 428 · QTIP 405** — et c'est cette
colonne-là qui compare des décodeurs.

**Le seul rapport inter-bras que ce run publie**, et il est mesuré dans un seul
processus sur les mêmes formes :

> **r = t(`Planes14`) ÷ t(QTIP) = 2,27× [2,27–2,28]** — plage entièrement
> au-dessus de 1. Le noyau à treillis va **2,27× plus vite que notre meilleur
> layout** sur ces formes. *(Le banc n'imprime pas `r` par round ; la plage est
> encadrée par l'extérieur à partir des deux intervalles — conservateur,
> **calculé**.)*

### L'erratum au pré-enregistrement tamponné, consigné parce qu'un `.ots` ne se réécrit pas

Le §5 du pré-enregistrement posait, « structurellement, quel que soit le
noyau », un plafond sur la fraction de borne d'octets convertie :
**f ≤ 59,6 %** dans ce processus. Mesuré : **f = 61,1 %** *(calculé sur les
médianes : 4,89 ÷ 8,00, la borne d'octets de 2,000 b/poids valant 16/2,000 =
8,00× ; contrôle sur les minima 405/661 = 61,3 %, même case)*, soit **1,5 point
au-dessus** contre un δ pré-défini de 0,2 point. Et la forme brute du même
fait :

> **t(QTIP) 2,246 ms < t(`nullk`) 2,306 ms** — séparation **2,7 %** contre une
> résolution **2R = 0,72 %**. Un bras qui lit 0,91 Go de poids finit avant
> notre passe qui n'en lit **aucun**.

La phrase tamponnée « aucun bras ne peut aller plus vite que `nullk` » est donc
**mesurée fausse**, et le journal la consigne comme telle plutôt que de la
retoucher. Ce qu'elle supposait sans le dire, c'est que la géométrie de
lancement était commune ; elle ne l'est pas. **`nullk` borne notre structure de
noyau — un warp par ligne de sortie, 252 lancements — et pas la machine.**

### Le mécanisme est structurel, et ce n'est pas un défaut d'implémentation

C'est la formulation que le papier retient (`paper/sections/layouts.tex`), et
elle mérite d'être ici parce qu'elle borne la **famille entière** plutôt qu'un
layout :

> Les deux formats stockent **2,000 bits de code par poids sur le disque**. Sur
> les mêmes matrices, le noyau à treillis lit **0,91 Go** là où `Planes14` en
> lit **2,18** — **2,40× d'octets** pour **2,27× de temps**, les deux noyaux
> convertissant **61 % et ~65 %** de leur borne d'octets *(calculé)*. À
> efficacité quasi égale, **l'écart de temps suit l'écart de trafic**, et ce qui
> fixe le trafic est le **dépliage au chargement**.
>
> Un codebook de **1,1·10¹⁴ points ne tient pas dans une table de
> correspondance**, là où un **état de treillis 16 bits** y tient (2 Kio ; les
> codebooks tabulés, comme l'E8P de QuIP#, tiennent 2¹⁶ entrées). **L'index de
> réseau doit donc être déplié** en un flux de plans de bits à **4,80 b/poids**,
> et le noyau paie ces octets à la vitesse de la mémoire.

🚨 **Ce n'est pas « notre décodeur est mal écrit ».** Le dépliage est imposé par
la **taille du codebook**, c'est-à-dire par ce qui fait la qualité du réseau de
Leech. Les quatre routes fermées sous `Planes14` (section précédente) sont
autant de tentatives de replier ce flux ; toutes se sont retrouvées bornées en
**calcul**. Le dépliage est le prix d'entrée de la famille, pas un poste
d'optimisation.

⚠️ **Trois réserves déclarées d'avance et non corrigées** (§3 du
pré-enregistrement) :

1. **QTIP ne porte ni queue f32 ni échelle de ligne f32** — en sa faveur, dans
   la comptabilité `b/poids noyau` de la table.
2. **Payload pseudo-aléatoire** : licite pour un code à débit fixe dont le noyau
   n'a aucune branche dépendante des données. 🚨 **Conséquence absolue :
   AUCUNE PHRASE DE QUALITÉ ne peut s'appuyer sur ce bras.** Il mesure un
   décodeur, jamais un modèle.
3. **QTIP tel que livré**, réglage `<<<128, 1024>>>` figé, **aucun tuning dans
   aucun sens** — ni chez lui, ni chez nous.

Contrôles du run : exactitude vérifiée ligne à ligne contre la référence f64 —
**1 105 920 lignes**, pire erreur QTIP **5,4e-8·Σ|w·x|** contre notre seuil de
1e-5 (pas le 1e-3 concédé à AWQ pour sa sortie binary16) ; **0 octet local** sur
les cinq shims (48–56 registres) ; dispersion QTIP **0,13 %** de la médiane ;
dérive inter-phases **R = 0,36 %**, donc tout écart sous **0,72 %** est déclaré
non résolu — et `r` est séparé de 1 par 127 %.

## L'échelle ne transfère pas d'une architecture à l'autre (A100, 2026-08-18 ; horloges lues le 2026-08-23)

🚨 **Tout « × vs FP16 » de ce document est un résultat L40S/Ada.** Ce n'était
jusqu'ici pas une réserve mais un angle mort : une seule carte avait jamais
tourné. Le lot F4 en a mesuré une seconde — **A100-SXM4-80GB**, 108 SM, L2
41,9 Mo lue, HBM2e — avec le **même code**, les mêmes bras, le même protocole
(7 rounds dont 2 jetés, rapports round par round), la compilation basculée en
`sm_80` par `LLVQ_NVRTC_ARCH`
([`mesures/f4-a100-2026-08-18.txt`](mesures/f4-a100-2026-08-18.txt),
job `6a8559fc…`, ~1,00 $ **estimé** — premier job `a100-large` du registre,
tarif jamais observé).

| bras | méd ms A100 | vs FP16 A100 | Go/s (min) A100 | repère L40S : vs FP16 · Go/s |
|---|---|---|---|---|
| plancher `nullk` | 4,107 | 1,68× [1,68–1,68] | 18 | 4,79× · — |
| FP16 (témoin maison) | 6,915 | 1,00× | 1052 | 1,00× · 661 |
| FP16 cuBLAS | 6,041 | 1,14× [1,14–1,15] | 1204 | 1,02× · 671-676 |
| AWQ w4g128 🏁 | 3,793 | 1,82× [1,82–1,82] | 501 | 3,37× · 584 |
| **`Planes14`** | 8,742 | **0,79× [0,79–0,79]** | 250 | 2,16× · 425-427 |
| `Slot32` | 9,413 | 0,73× [0,73–0,73] | 266 | 1,87× · 428 |
| `Planes12x` | 9,423 | 0,73× [0,73–0,73] | 209 | 1,98× · 356 |
| `Golay70` v2 | 11,121 | 0,62× [0,62–0,62] | 147 | 1,77× · 263 |
| `Golay70` v1 | 15,705 | 0,44× [0,44–0,44] | 104 | 1,31× · 195 |

🚨 **Les `×` inter-cartes NE SE DIVISENT PAS** — règle posée au §3 du
pré-enregistrement, et la colonne de droite est un **repère**, pas un
dénominateur. Deux cartes, deux processus, deux témoins FP16 différents.

**Sur A100, aucun bras à décodage ne bat le FP16.** L'exactitude, elle, ne
bouge pas : la vérification f64 rend des pires erreurs **identiques** à celles
du L40S (2,2e-8 pour `Slot32` et `Planes14`, …) — *l'arithmétique des noyaux ne
dépend pas de la carte, seule leur conversion octets→temps en dépend.*

**Trois choses que la table établit, toutes mesurées.**

1. **Le FP16 convertit la HBM, nous non.** Le témoin passe de 661 à
   **1 052 Go/s** et cuBLAS de 672 à **1 204** ; pendant ce temps les Go/s
   effectifs de nos bras **chutent** : `Planes14` 425 → **250**, `Slot32`
   428 → **266**, `Planes12x` 356 → **209**, `Golay70` v2 263 → **147**. Une
   borne mémoire ne produit pas ça : **sur A100, ces noyaux sont bornés par le
   calcul par SM.**
2. **Le plancher bouge avec la carte**, ce qui est la démonstration la plus
   directe qu'il n'est pas une propriété du problème : 2,305 → **4,107 ms**, et
   il ne vaut plus que **1,68× FP16** contre 4,79×. Le sol
   latence/lancement mange **59 %** du temps FP16 sur A100 contre 21 % sur
   L40S.
3. **L'ordre INTERNE des bras LLVQ tient**, à une égalité près : `Planes14`
   devant, `Slot32` et `Planes12x` à 0,73× tous deux — sur L40S `Planes12x`
   devançait `Slot32`. **C'est l'échelle CONTRE FP16 qui s'inverse en bloc**,
   pas la hiérarchie de nos layouts entre eux.

### Le lot G tranche l'hypothèse : c'est le rapport d'horloges

F4 laissait une **hypothèse candidate explicitement non tranchée** — « les
facteurs ×1,6-1,8 sont compatibles avec le rapport des fréquences SM
constructeur » — en refusant de publier un pic que personne n'avait lu. Le lot
G l'a lu, **pendant le banc**, à 1 Hz
([`mesures/g-horloges-planes12x-2026-08-23.txt`](mesures/g-horloges-planes12x-2026-08-23.txt),
jobs `6a8c1427…` / `6a8c1428…`, 1,00 $ pour le lot) :

| carte | driver | SM médiane | SM max | événement d'horloge |
|---|---|---|---|---|
| L40S | 580.178.04 | **2 520 MHz** | 2 520 MHz | `0x1` = GpuIdle seul |
| A100 | 580.159.03 | **1 410 MHz** | 1 410 MHz | `0x1` = GpuIdle seul |

**Les deux cartes tournent épinglées à leur boost max, sans aucun bridage
thermique ni de puissance.** Le rapport **2 520/1 410 = 1,787** *(calculé)*
tombe dans le critère `[1,60 ; 1,95]` posé d'avance et **colle au ralentissement
mesuré du témoin sans lecture** : `nullk` **×1,772** au banc G, **×1,781** au
banc F4 publié. **Le ×1,78 de la table A100 EST le rapport d'horloges.**

⚠️ **C'est une preuve d'HORLOGE, pas un profil d'occupation.** Les compteurs qui
trancheraient le reste sont **refusés par la plateforme** (`ERR_NVGPUCTRPERM`,
constaté en F3 : `ncu` s'installe et s'attache, la carte refuse les compteurs).
Indisponibilité déclarée comme **fait de plateforme**, sans retentative.

**Conséquence de rédaction, et elle vaut pour tout ce dossier** : le claim « le
décodage tourne à la vitesse du matvec » cesse d'être une propriété du format
pour devenir un **résultat L40S/Ada à domaine de validité mesuré**. Le point
A100 le **borne**, il ne l'étend pas — et c'est plus fort qu'un point unique
assorti d'une limitation « une seule carte » : la conversion octets→temps est
une propriété du **couple format × carte**, pas du format.

🔎 **Ce que F3 apprend au passage, et qui affaiblit une hypothèse sans la
réfuter** (0,86 $,
[`mesures/f3-events-2026-08-19.txt`](mesures/f3-events-2026-08-19.txt)) : le
chronométrage par **events CUDA** rend un écart hôte−device de **0,1–0,2 %**,
soit **4 à 8 µs par round entier**, deux ordres de grandeur sous l'attente du
pré-enregistrement. Dans ce banc, la soumission hôte est **entièrement
recouverte**. Ce que ça élimine est **une** hypothèse — « le poste latence,
c'est l'hôte qui n'arrive pas à suivre » — pas le poste lui-même, qui reste
**device** : bulles inter-noyaux, montée en occupation, et ces écarts-là sont
**dans** le span device. ⚠️ Deux dénominateurs, encore : F3 mesure **le banc**
(252 matrices enfilées), l'attribution 39/33/19 du 2026-08-05 mesure **le chemin
modèle** (2,04 ms/token, un autre dispatcher).

🔎 **Et le dénominateur FP16 lui-même a été vérifié** (F1, 0,08 $,
[`mesures/f1-cublasf16-2026-08-18.txt`](mesures/f1-cublasf16-2026-08-18.txt)) :
`r = médiane(t_témoin ÷ t_cuBLAS)`, formé round par round dans un même
processus, vaut **1,024** à deux bras et **1,015** à cinq, contre un seuil de
1,05 posé d'avance. **Notre témoin FP16 maison est au niveau de cuBLAS sur
L40S** — donc tous les rapports « vs FP16 » publiés ici tiennent, et ils ne
flattent pas le numérateur. ⚠️ Verdict **L40S** : sur A100 le même témoin est à
**1,14×** de cuBLAS, ce qui ne le contredit pas — autre carte.

## Le mur de la partagée — la seule borne de ce document qu'aucun format ne déplace (2026-08-17)

Tout ce qui précède oppose des **layouts**. Cette section n'en oppose aucun :
elle porte sur le **noyau de rotation**, que tous les layouts appellent à
l'identique, et elle dit à quelle taille de modèle le chemin fusé s'arrête —
indépendamment du format des poids.

**La raison est structurelle et tient en deux phrases.** Une transformée de
Walsh–Hadamard est `log₂ m` étages séparés par des **barrières**, et CUDA n'a
pas de barrière entre blocs. Donc `rot_apply` est un noyau à **un bloc**, qui
met **toute l'activation en mémoire partagée** — une f32 par coordonnée, quel
que soit le dtype d'entrée (le noyau élargit la f16 au chargement). La largeur
du modèle devient une contrainte matérielle dure : `rotate.cu` l'assume et
l'écrit depuis le premier jour, en nommant le 32B comme le cas qui ne rentre
pas.

Ce que personne n'avait vu, c'est qu'il y a **deux** bornes, pas une, et que
le garde comparait à la mauvaise :

| attribut CUDA | L40S | qui le lisait |
|---|---|---|
| `MAX_SHARED_MEMORY_PER_BLOCK` | **49 152 o** | le garde, et lui seul |
| `MAX_SHARED_MEMORY_PER_BLOCK_OPTIN` | **101 376 o** *(mesuré)* | personne, jusqu'au préflight du 2026-08-17 (soir) |
| `MAX_SHARED_MEMORY_PER_MULTIPROCESSOR` | 102 400 o | le préflight, à l'affichage |

La seconde s'obtient en posant `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`
sur la **fonction**, une fois, avant tout lancement. Le noyau ne change pas
d'une ligne.

> 🆕 **La ligne d'opt-in portait « fiche sm_89, pas encore lue sur carte » ;
> c'était vrai le jour où elle a été écrite, et le préflight du soir même l'a
> périmée.** Elle est **mesurée** sur la carte qui a servi le 14B — `101 376 o`,
> soit exactement ce que la fiche annonçait
> ([`mesures/fusedrun-14b-2026-08-17.txt`](mesures/fusedrun-14b-2026-08-17.txt),
> étape 1, job `6a83121be55292eada79b611`). ⚠️ Le verdict du 32B **ne bouge pas
> d'un octet** : 25 600 × 4 = 102 400 o demandés contre 101 376 offerts, il
> manque **1 024 o** — ce qui change, c'est que ce « manque » cesse de reposer
> sur une plaque signalétique. Les 102 400 o/SM ne le sauvent pas : c'est un
> budget **par SM**, pas la limite d'un **bloc**.

**Les quatre largeurs du délivrable** — `intermediate_size`, l'entrée de
`down_proj`, la plus large activation que la rotation ait à mettre en partagée :

| modèle | `intermediate_size` | partagée | défaut 49 152 | opt-in 101 376 *(mesuré)* |
|---|---|---|---|---|
| 4B | 9 728 | 38 912 | ✅ | ✅ |
| 8B | 12 288 | **49 152 — exactement la limite, à l'octet** | ✅ | ✅ |
| **14B** | 17 408 | 69 632 | ❌ | ✅ |
| 32B | 25 600 | 102 400 | ❌ | ❌ **de 1 024 o** |

🚨 **Le 8B tenait pile sur la borne, et rien ne pouvait le dire.** Un garde ne
parle que lorsqu'il refuse ; celui-ci a laissé passer 12 288 × 4 = 49 152 sans
un mot, et le premier modèle plus large que le 8B est donc le premier à taper
le mur — d'emblée, et sur un job facturé.

**Le job, parce qu'un échec propre sur un garde est une mesure**
([`mesures/rot-partagee-14b-2026-08-17.txt`](mesures/rot-partagee-14b-2026-08-17.txt)) :
`6a82f40ce55292eada79b526`, L40S, **0,24 $**, exit 1 après 488 s, aucun token
produit. Message : *« rotation de largeur 17408 : 69632 o de partagée demandés,
la carte en offre 49152. Le noyau à un bloc ne convient pas à cette largeur
(cf. rotate.cu). »* Le garde a fait ce qu'il annonçait — refuser plutôt que
corrompre — sur une borne qui n'était pas la bonne.

**Ce qui a été corrigé** : l'attribut d'opt-in est **lu** (jamais dérivé d'un
`shared_per_sm − 1024`, qui est une supposition sur une réserve appartenant au
driver), il est posé sur la fonction au chargement, et le garde compare aux
**deux** bornes en nommant celle qui est franchie. L'arithmétique de la
décision vit dans `llvq_cuda::shared`, **portable et testée sur le Mac** — ses
deux appelants sont sous `cfg(linux)`, où seul un build d'image les compile
(§3 de la passation du 2026-08-16, trois casses en deux jours).

⚠️ **Le 32B reste refusé, et ce refus est le point critique** : il manque
l'opt-in de **1 024 o**, la réserve du driver, et un garde qui le laisserait
passer produirait la corruption silencieuse que `rotate.cu` annonce
explicitement. La seule piste nommée — **et non tranchée, ni conçue ici** —
est de stager l'activation en **f16** : 51 200 o, sous le défaut, opt-in même
pas nécessaire. Son coût n'est pas une astuce mais un arbitrage numérique :
à `n = 25 600` la transformée est `m = 1 024`, soit **10 barrières de
Walsh–Hadamard**, suivies d'un mélange dense `k = 25` — 25 600 termes
contribuent à chaque coordonnée, la profondeur d'accumulation d'une Hadamard
de **~14,6 étages**, qui se ferait alors en demi-précision. `rotate.cu` nomme
de son côté une autre issue, le découpage en deux noyaux, tout aussi non
écrite.

**Deux choses que ce mur n'est pas.** Ce n'est pas un mur de **format** : la
rotation est identique sous `Planes14`, `Planes12x`, `Slot32`, `Golay70` et
E1v, et aucun verdict des sections précédentes ne le déplace d'un octet. Et ce
n'est pas une part du **plancher** du 2026-08-16, qui chronomètre 252
projections d'un token, rotation exclue — deux dénominateurs, encore, et les
rapprocher demanderait de refaire l'attribution.

## `Planes12x` change d'état : mesuré SERVI au 4B (G3, 2026-08-23)

Ce document a nommé trois états — **absent** (aucun code ne le sélectionne),
**câblé** (`LLVQ_FUSED_LAYOUT` l'accepte, donc il est *mesurable*), **servi**
(c'est le défaut, donc ce que rendent les chiffres publiés). `Planes12x` était
« câblé, non servi » depuis le 2026-08-09, avec une réserve explicite : *« câblé
n'est toujours pas mesuré »*, le noyau f16 n'étant alors que **compilé**.

**La réserve est levée : il a tourné bout-en-bout**
([`mesures/g-horloges-planes12x-2026-08-23.txt`](mesures/g-horloges-planes12x-2026-08-23.txt),
job `6a8c2355…`, L40S, 0,79 $ ; protocole `fusedrun` — 1 génération jetée +
5 chronométrées, médiane [plage], 128 tokens, tokens comparés au bras dense ;
`LLVQ_FUSED_LAYOUT=planes12x LLVQ_EMBED=q8`) :

| bras | tok/s (médiane) [plage] | Go carte (**compte d'octets hôte**) | b/poids **inférence** |
|---|---|---|---|
| **fusé `Planes12x` + q8** | **85,0 [84,7–85,1]** | **2,36** (proj 1,94 + portés 0,41) | **4,277** |
| dense f16 | 43,4 [43,4–43,4] | 8,04 | 16,000 |

**×1,96 [1,95–1,96]** de vitesse *(quotient des médianes — les rounds des deux
bras ne coexistent jamais, chaque bras charge son modèle exclusivement, donc
aucun rapport round par round n'existe : c'est la forme de `fusedrun`, pas celle
des bancs à bras entrelacés)*, **÷3,41** de mémoire carte, et **divergence
gloutonne au token 89/128** — le tie-break historique de `Planes14`, reproduit.
Les deux prédictions du pré-enregistrement tiennent : débit ~84 ∈ [76 ; 90],
VRAM ~2,39 ∈ [2,30 ; 2,48].

**Contre `Planes14` servi dans la même comptabilité** (B2, 87,0 tok/s, 2,56 Go
hôte) : **−2,3 % de débit pour −0,20 Go de carte**. C'est **le point servi le
plus compact mesuré** du dépôt.

⚠️ **Trois choses que ça ne dit pas.**

1. **`Planes12x` n'est pas devenu le défaut**, et rien ici ne le propose. Son
   état exact est « **mesuré servi, non défaut** » — le quatrième de la
   nomenclature, et le replier sur « servi » ferait croire que les chiffres
   publiés sortent de lui.
2. **Le transcodage coûte** : **1 340 s de chargement**, contre ~131 s pour
   `Planes14` au même protocole — la recherche réseau à 5 niveaux par bloc. Coût
   **hors ligne**, payé une fois, mais payé sur carte louée.
3. **Le −2,3 % n'est pas un rapport de banc.** Il vient de deux jobs, deux
   chargements, deux médianes ; le banc à dix bras, lui, place `Planes12x` à
   **0,93× [0,93–0,93]** de `Planes14` sur un token de projections. Deux
   dénominateurs — un token complet contre 252 projections — **et l'écart entre
   les deux est exactement ce que le reste du modèle amortit.**

## Note de provenance

**Le fichier.** « 2,1595 b/poids » = bits de payload / *tous* les poids (queue
comprise), imprimé par `bin/seal` ; « 2,1696 » = mêmes bits / poids
*quantifiés* seuls, imprimé par `bin/smoke`. Deux dénominateurs, pas deux
mesures. Le chiffre du fichier pesé (981 Mo) est cohérent avec les deux.

**La RAM — trois comptabilités, à ne jamais aligner sans étiquette.** C'est
la faute que le lot K-1(a) est venu supprimer, et qu'on peut refaire :

| comptabilité | ce qu'elle compte | `Slot32` sur le 4B |
|---|---|---|
| `rtbits` / `arrbits` | payload + une base u32 par groupe, stride arrondi à l'octet | **5,3756** |
| `bin/matvec` (une couche) | idem, sur gate_proj seule | **5,375** |
| `bin/thesis` (modèle entier) | idem **+ queue f32 + échelles de ligne f32** | **5,510** |

Les deux premières lignes coïncident parce que c'est la même comptabilité ; la
troisième diffère parce que son numérateur est plus large. Aucune n'est
fausse — c'est de les mélanger dans une seule échelle qui l'était.

🆕 **Il y en a une QUATRIÈME, et elle est arrivée sans être nommée : la
comptabilité INFÉRENCE.** Le chemin servi ne facture pas sa queue comme le banc
— `fusedrun` la porte en **binaire16** (`TailPolicy::KeepExact` à 16 bits) là où
le banc la facture en **f32**. Les deux valeurs sont donc **structurellement
différentes**, et aucune n'est fausse :

| layout, sur le 4B | `b/poids` **noyau** (banc, queue f32) | `b/poids` **inférence** (`fusedrun`, queue binaire16) |
|---|---|---|
| `Planes14` | **4,804** | **4,729** |
| `Planes12x` | **4,342** | **4,277** |

*(Sources : banc à dix bras du 2026-08-21 ; `fusedrun` de B2 pour `Planes14`
(2026-08-18) et de G3 pour `Planes12x` (2026-08-23), qui impriment l'étiquette
eux-mêmes.)* ⚠️ **Un « 4,804 » et un « 4,729 » qui se suivent dans une phrase ne
sont pas une dérive de mesure, ce sont deux numérateurs** — exactement le motif
que cette note existe pour empêcher. La règle est inchangée : **étiqueter, ou se
taire**.

**Et une note de comptabilité mémoire, du même ordre** : les « Go carte » de ce
document sont partout un **compte d'octets hôte** imprimé par `fusedrun`, jamais
une lecture de `nvidia-smi`. C'est le même instrument des deux côtés de chaque
rapport, donc les ÷ sont licites ; ce sont les **valeurs absolues** qui ne se
comparent pas à un affichage carte. 🕳️ Le « 2,60 Go » du 4B q8 qui circule dans
le dossier était l'**arrondi carte d'époque** ; le compte hôte du même bras rend
**2,56 Go** (B2, 2026-08-18) — 1,6 % d'écart, deux instruments, aucune
régression.
