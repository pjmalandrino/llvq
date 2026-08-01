# Format de noyau — mesures, impasses, architecture (2026-07-31)

> Branche `g6-format-noyau`. Tout chiffre ci-dessous est mesuré par un banc
> reproductible (`decbench`, `decprofile`, `classprofile`, `arrbits`,
> `decfast`, `decfull`), et l'ensemble a été passé au crible d'un audit
> adversarial de 5 agents (lecture seule) dont les corrections sont intégrées.

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

Le décodage Slot32 coûte **12 µs au-dessus du sol** — il se glisse dans les
bulles de latence mémoire, ce qu'un noyau fusé doit faire. L'échelle des
échanges bits↔vitesse est maintenant mesurée de bout en bout : 2,16 b/poids
(archive, indécodable) → 3,35 (nested, 0,68×) → 4,54 (Flat32, 0,90×) →
**5,375 (Slot32, 2,21×)**. Sur le 4B : ~2,44 Go de linéaires + 0,78 de
lm_head ≈ 3,2 Go/token → plafond ~124 tok/s, ~2,5× le FP16 — cohérent avec
le 2,2× mesuré sur la couche.

⚠️ Piste ouverte pour reprendre des bits sans perdre la vitesse : les blocs
à 5 niveaux (3,4 %) fixent le stride de ~2/3 des groupes à 17 octets ;
plafonner le quantifieur à L ≤ 4 (ou isoler les blocs L=5) ramènerait le
stride typique à 14 octets, ~4,4 b/poids, sans toucher au décodeur.

## La thèse, sur le modèle entier (2026-08-01)

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

### Le trou de couverture que ça a fermé

`bin/matvec` stageait toute l'activation en mémoire threadgroup : 10 Ko à
d_in = 2560, mais **38 Ko à d_in = 9728 contre la limite Metal de 32 Ko**. Les
36 `down_proj` du modèle n'auraient pas pu tourner — le 2,2× était mesuré sur
une forme qui passe. Les deux noyaux de `thesis` tuilent l'activation par
128 blocs (3072 colonnes, 12 Ko), donc les six formes du modèle empruntent le
même code.

### Le prix en RAM, et que c'est un cadran

Sur le modèle entier `Slot32` coûte **5,51 b/poids** (contre 5,375 sur
gate_proj : les autres formes ont d'autres distributions de classes et
d'autres arrondis de stride). Le fichier, lui, ne bouge pas — 2,1696 b/poids.
D'un même `.llvq` on charge le format qu'on veut :

| en RAM | b/poids | projections | vitesse |
|---|---|---|---|
| `Grouped32` | 3,35 | 1,52 Go | 0,68× |
| `Flat32` | 4,54 | 2,06 Go | 0,90× |
| **`Slot32`** | **5,51** | **2,50 Go** | **2,07×** |

Même au plus large, le modèle chargé fait ~3,3 Go contre 8,045 en FP16.

### Ce que ce chiffre ne couvre pas

Attention, normes, activations et la rotation de `x` ne sont pas mesurées —
seulement la part que la quantification change. Et **le noyau n'est pas branché
dans `bin/run`** : le runner livré décode toujours en mémoire puis fait un
matvec ordinaire. C'est le dernier chantier.

⚠️ Correction au passage : l'estimation « ~2,6-3,0 b/poids » de la table
d'architecture ci-dessus ne comptait ni la classe, ni l'adressage. Le vrai
plancher adressable est 3,35 ; les plafonds deviennent 174/154 tok/s, toujours
~3,1-3,5× le FP16, et le lm_head pèse toujours le tiers du trafic.

## Note de provenance

« 2,1595 b/poids » = bits de payload / *tous* les poids (queue comprise),
imprimé par `bin/seal` ; « 2,1696 » = mêmes bits / poids *quantifiés* seuls,
imprimé par `bin/smoke`. Deux dénominateurs, pas deux mesures. Le chiffre
du fichier pesé (981 Mo) est cohérent avec les deux.
