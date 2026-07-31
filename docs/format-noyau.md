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

## Note de provenance

« 2,1595 b/poids » = bits de payload / *tous* les poids (queue comprise),
imprimé par `bin/seal` ; « 2,1696 » = mêmes bits / poids *quantifiés* seuls,
imprimé par `bin/smoke`. Deux dénominateurs, pas deux mesures. Le chiffre
du fichier pesé (981 Mo) est cohérent avec les deux.
