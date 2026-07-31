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
| noyau | **un bloc par lane**, décodage sériel des masques (~100-150 lane-ops/bloc) | sous le point de bascule (~185-200) | à mesurer sur Metal |

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

## Ce que le micro-banc Metal doit mesurer (premier travail local)

1. Décodage masques, un bloc par lane, layout SoA — lane-ops réels, divergence,
   pression registre, coalescence. C'est LE chiffre qui décide.
2. En contrôle : décodage direct du rang (recurrence u64, division par magie)
   pour vérifier le ~509.
3. Les risques non chiffrés : formes de classes variables entre lanes
   (divergence), 24 poids × 32 blocs en vol (registres).

## Note de provenance

« 2,1595 b/poids » = bits de payload / *tous* les poids (queue comprise),
imprimé par `bin/seal` ; « 2,1696 » = mêmes bits / poids *quantifiés* seuls,
imprimé par `bin/smoke`. Deux dénominateurs, pas deux mesures. Le chiffre
du fichier pesé (981 Mo) est cohérent avec les deux.
