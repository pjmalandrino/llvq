# Rapport — lot A : le noyau fusé en inférence, et la confrontation au 4 bits

> **2026-08-06.** Ce document rapporte ce qui a été mesuré et ce que ça
> établit. Il est distinct de
> [`passation-lot-a-2026-08-06.md`](passation-lot-a-2026-08-06.md), qui dit où
> reprendre — celui-ci dit ce qu'on sait.
>
> Périmètre : le lot A de [`spec-lot-a-2026-08-05.md`](spec-lot-a-2026-08-05.md),
> quatre étapes, quatorze jobs sur `l40sx1`, **~2,2 $** (somme des coûts
> rapportés par le moniteur). Toutes les millisecondes brutes sont dans
> `docs/mesures/`.

---

## 1. En une page

Le noyau fusé existait depuis des semaines et **n'était appelé nulle part**.
Le modèle décodait ses poids au chargement et tournait exactement comme un
modèle f16 — le protocole miniature l'avait mesuré sans détour : 42,7 tok/s
contre 42,8, aucun bénéfice d'aucune sorte.

Le lot A l'a branché, mesuré, et confronté au 4 bits.

**Ce que le branchement produit :**

| | avant | après |
|---|---|---|
| mémoire carte | 8,04 Go | **3,28 Go** (÷2,45) |
| débit | 42,7 tok/s | **47,0** (×1,08) |
| chargement | 209 s | **128 s** (÷1,47) |
| texte | référence | identique sur 88 tokens |

**Ce que la confrontation dit :**

| bras | b/poids | ppl | MMLU micro |
|---|---|---|---|
| f16 | 16,000 | 12,2369 | 70,32 % ± 1,28 |
| AWQ 4 bits (officiel Qwen) | 4,156 | 13,5207 | 70,04 % ± 1,25 |
| **LLVQ 2 bits (nous)** | 2,170 | 16,9422 | **55,59 % ± 1,35** |

**Sur un Qwen3-4B, le 4 bits nous domine en perplexité, en capacités et en
mémoire vive.** Notre seul avantage est le fichier sur disque — 1,77 Go contre
2,26 — et le disque n'a jamais été ce qui limite.

Ce qui survit est le **noyau** : un décodeur de réseau de Leech fusé qui bat le
f16 de 1,89× sur les projections, tourne dans un vrai modèle et rend les mêmes
tokens. Ça n'existe nulle part ailleurs, papier compris. Ce qui ne survit pas,
c'est le **produit** sur un modèle de cette taille.

---

## 2. A1 — le premier passage sur carte

Cinq critères de sortie, tous verts.

**L'oracle.** `max |Δhidden| = 0.000e0` — zéro exact, pas « proche de zéro »,
et identique aux passages précédents sur ce backend. La passe avant maison
reproduit celle de `candle-transformers`, donc tout écart mesuré ensuite est
imputable au chemin fusé et non au harnais. C'était le gate absolu de la spec
et il avait été sauté ; il coûte 42 secondes et un centime.

**La rotation sur carte.** 47 registres, **zéro octet de débordement**, et la
carte rend *exactement* les chiffres du harnais Mac (9,523·10⁻⁸ en relatif,
6,234·10⁻⁷ sur la pire coordonnée). Cette égalité est vérifiable a priori :
chaque coordonnée de sortie est calculée par un seul thread, sans réduction
inter-thread — rien ne peut réassocier. C'est ce qui distingue ce noyau du
matvec, dont les sommes Metal et CUDA divergent par construction.

**L'égalité des tokens.** Aucune divergence sur 32 tokens ; divergence au
**token 89 sur 128**. La grille de lecture était posée d'avance : « 1 à ~5 =
bug, tardive = tie-break ». 88 tokens gloutons identiques d'affilée à travers
252 projections × 36 couches, deux fois par token, puis deux candidats à un
arrondi l'un de l'autre et l'argmax tranche autrement. Les deux suites restent
grammaticales et de même qualité.

Exiger le contraire reviendrait à exiger la bit-exactitude entre deux ordres
d'accumulation différents, ce qui n'est pas sur la table.

**Les octets et le débit.** 3,28 Go contre 8,04 ; 47,0 tok/s contre 43,5. Le
chiffre de mémoire était calculé depuis des mois — il est maintenant mesuré, et
il retombe exactement dessus.

---

## 3. La plomberie — de 41,7 à 47,0 tok/s

Le premier run complet a donné **41,7 tok/s, soit 4 % plus lent que le chemin
dense**, alors que le banc mesure les mêmes 252 projections à 5,84 ms contre
10,99 en f16. Le gain du noyau était réel et intégralement rendu par la couture.

Le diagnostic s'est fait en comptant les lancements, puis chaque poste a été
mesuré séparément :

| poste | prévu | mesuré |
|---|---|---|
| 252 allocations + 504 remises à zéro | — | **−1,25 ms** |
| 252 conversions de type en sortie | ~1,3 ms | **−0,80 ms** |
| 252 préparations en entrée | ~1,3 ms | **0 — n'existait pas** |
| 108 lancements (regroupement q/k/v) | 0,8 ms | non fait |

Deux enseignements de méthode :

**Le troisième poste n'existait pas.** `to_dtype` et `contiguous` retournent un
simple clone quand la conversion est l'identité (`tensor.rs:2453` et `:2466`),
et l'activation arrive déjà au bon format. Le chiffre venait d'un comptage de
lancements, pas d'une lecture du code — exactement l'erreur que l'audit
reprochait au compte d'instructions niveau source.

**Le partage du tampon est sûr, mais pas grâce au verrou.** Celui-ci est
relâché bien avant que le moindre noyau ne tourne, les lancements étant
asynchrones. Ce qui rend le partage sain, c'est que les deux noyaux vont sur le
**même flux**, qui exécute dans l'ordre d'émission. Sur deux flux séparés, ce
serait une course qui se manifeste une fois sur cent.

---

## 4. L'attribution du gisement — où part le temps du noyau

Avant le branchement, le noyau lisait 2,50 Go à 430 Go/s là où le témoin f16
en tire 662 de la même carte : ~2 ms entre le plancher physique et le mesuré.
L'audit de perf listait quatre candidats aux bornes qui se recouvrent.

Trois noyaux « sol », chacun ajoutant *une* chose que fait le vrai noyau, ont
tranché :

| poste | Δ ms | part |
|---|---|---|
| latence non masquée / occupation | 0,803 | **39 %** |
| flux Slot32 (motif de lecture) | 0,681 | 33 % |
| décodage résiduel | ~0,396 | 19 % |
| lectures d'activations (conflits de bancs) | 0,118 | 6 % |
| gather de la table | 0,041 | 2 % |

**Trois prédictions de l'audit sont renversées.** Les conflits de bancs valaient
0,3 à 2,9 ms selon lui, avec le padding remonté en priorité n°1 : ils valent
0,118 ms. Le gather était surestimé d'un facteur 12. Et l'ALU, déclarée « hors
de cause », était le premier poste — jusqu'à ce que la fusion montre que les
deux tiers en étaient de la latence.

**La fusion q+k+v / gate+up rend 0,803 ms** (13,8 % du noyau), sortie
**identique au bit près** sur 921 600 lignes, et **octets lus identiques à
0,00 %** — donc le gain est purement géométrique. `k_proj` et `v_proj` seules
lançaient 128 blocs sur les 852 que la carte tient, rendaient 157 Go/s contre
469 pour une forme pleine, et leur rapport contre le f16 était de 1,06× : rien,
pour 13 % du temps.

---

## 5. A2 et A3 — deux portes fermées, chiffrées

**A2 — la recopie du cache.** Fermée par candle, tranchée sans dépenser un
centime : `broadcast_matmul` matérialise aussi (le `TODO: Avoid concretising`
est de candle), `repeat_kv` est la forme que ses auteurs ont mesurée comme la
plus rapide, et boucler sur les têtes ajouterait 288 lancements par token.

Le chiffrage change la lecture : ~0,06 ms à 70 tokens de contexte (0,3 %,
invisible dans nos bancs) mais **~3,6 ms à 4096** — plus que tout ce que la
plomberie a récupéré. **À rouvrir dès qu'un chiffre à contexte long est visé.**

**A3 — CUDA Graph.** Mesuré, négatif.

```
stream legacy   3,63 µs/lancement   →  0,915 ms sur 252
stream frais    3,67                →  0,926
frais + graph   2,97                →  0,748
```

`g = 3,63 µs`, et **trois instruments indépendants concordent** : 1,85 µs de
soumission processeur, 3,3 µs bout-à-bout sur un noyau minuscule, 3,63 ici. La
fourchette [1 ; 4] µs que le dossier refusait de trancher est fermée.

ε = 0,915 ms, 15,8 % du bras LLVQ : le terme est réel, l'audit avait raison. Mais
**le graphe n'en récupère que 18 %** — il supprime le trafic vers le pilote, pas
la mise en route des blocs sur les processeurs. Soit **0,8 % d'un token**, et
c'est un plafond. A3(b) était conditionné à un (a) concluant : il ne l'est pas.

Effet de bord instructif : changer de flux **coûte** (3,67 contre 3,63), parce
que cudarc passe alors en mode multi-flux et insère deux événements par
lancement. C'est aussi ce qui invalidait la capture au premier essai.

---

## 6. A4 — la campagne, et le verdict

Même harnais, même carte, **même empreinte de tokens des deux côtés** —
`3f1baca9033bf251` en perplexité, `65dcd53655e8bfa5` en MMLU. C'est la
condition de comparabilité, et elle est vérifiée plutôt que supposée.

**Le harnais est certifié.** La baseline rend 70,32 % contre les **70,42
exigés** par la spec : 0,10 point, 0,08 σ. Et 12,2369 de perplexité contre
12,2361 publié. Rien n'a bougé, donc les deux autres bras sont lisibles.

**Le 4 bits ne perd rien.** −0,28 point de MMLU, très en dessous de l'erreur
d'échantillonnage (±1,25). Sur cet axe, l'AWQ officiel de Qwen est
**indiscernable du f16**.

**Nous perdons 14,73 points.** Le chiffre recoupe le −14,33 mesuré le
2026-08-02 sur le même fichier : la dégradation se reproduit, ce n'est pas un
accident de protocole.

### Le tableau complet

| | disque | VRAM | ppl | MMLU |
|---|---|---|---|---|
| f16 | 8,04 Go | 8,04 Go | 12,24 | 70,32 % |
| AWQ 4 bits | 2,26 Go | ~4,50 b/poids | 13,52 (×1,105) | 70,04 % (−0,28) |
| LLVQ, chemin dense | 1,77 Go | 8,04 Go | 16,94 (×1,384) | 55,59 % (−14,73) |
| **LLVQ, noyau fusé** | 1,77 Go | **3,28 Go** | idem | idem |

La dernière ligne est le seul endroit où le projet gagne quelque chose — et il
gagne **contre lui-même**, pas contre l'AWQ.

### Les réserves, et elles ne sauvent rien

* Notre harnais mesure la **reconstruction** de l'AWQ, pas son arithmétique
  fusionnée : il est chargé déquantifié, donc les octets qu'il occupe *ici* ne
  veulent rien dire. Cette réserve joue **contre** nous : elle ne concerne que
  la mémoire, axe où il est déjà devant.
* 2 280 questions sur 14 042, soit 16,2 %. Le ± est l'erreur d'échantillonnage
  seule. Un écart de 14,73 points ne s'explique pas par 1,3 point d'incertitude.
* Un seul modèle. Le 8B se dégradait moins (×1,267 contre ×1,386) : la piste
  d'échelle reste ouverte et non mesurée sur l'axe MMLU.

---

## 7. Ce que ça coûte, et ce que ça a appris sur la méthode

**~2,2 $ sur quatorze jobs**, contre 5,5 $ estimés pour la seule campagne. Dont
**0,34 $ perdus** en trois erreurs de couture entre candle et nos noyaux —
aucune dans le noyau lui-même :

1. un prefill refusé — toute génération commence par passer le prompt entier,
   ce que le garde-fou n'anticipait pas ;
2. `contiguous_offsets` lu comme un couple (offset, longueur) alors qu'il rend
   un intervalle (début, fin) — juste à l'offset 0, faux partout ailleurs ;
3. le tampon de rotation réalloué 252 fois par token.

Trois leçons transportables :

**Le bras qui peut échouer charge en premier.** Les deux premiers échecs ont
payé 209 secondes de chargement dense avant d'atteindre le code qui plante.
Quand un job est facturé à la minute, l'ordre n'est pas neutre.

**Rapporter l'erreur du corps avant celle de la fermeture.**
`CUDA_ERROR_STREAM_CAPTURE_INVALIDATED` signifie « une opération précédente a
échoué » — reporter ça au lieu de la cause coûte un job entier pour apprendre
que quelque chose s'est mal passé, sans dire quoi.

**Le build d'image prend 12-14 minutes, pas 40-70.** Sept builds, aucun échec.
Le dossier annonce un chiffre faux d'un facteur 4, et c'est précisément lui qui
décourageait d'écrire du code hôte instrumenté — donc lui qui a laissé le noyau
sans instrumentation jusqu'à ce lot.

---

## 8. Ce qui reste, par ordre de ce que ça décide

**Le seul chantier qui change le statut du projet : descendre les 5,51 b/poids
en VRAM sous les 4,50 du 4 bits.** Tant qu'on est au-dessus, le ÷2,45 mesuré
ici ne fait tenir aucun modèle là où le 4 bits ne tient pas — ce qui est la
seule raison d'aller à 2 bits. La piste identifiée est C1, les plans binaires
([`pistes-format-vram-2026-08-05.md`](pistes-format-vram-2026-08-05.md)) : le
format range l'appartenance de chaque coordonnée en *one-hot* là où 2 à 3 bits
suffiraient, et rien dans le décodeur ne consomme cette forme.

**Les leviers de vitesse restants, chiffrés et sans surprise** : le regroupement
q/k/v (0,8 ms, demande de changer la passe avant) et la rotation (1,52 ms, ne
disparaît qu'en la fusionnant dans les quatre producteurs d'activations).
Plafond visé ~51 tok/s, soit ×1,17 — utile, pas décisif.

**La recopie du cache**, à rouvrir pour tout chiffre à contexte long.

**L'axe d'échelle**, jamais mesuré sur MMLU : le 8B se dégrade moins que le 4B
en perplexité. Si la tendance tient à 70B, le 2 bits redevient nécessaire là où
le 4 bits ne rentre plus. C'est la seule hypothèse qui puisse renverser le
verdict de la §6, et elle n'est pas testée.
