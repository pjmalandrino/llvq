# Face au 4 bits — la comparaison qui manquait (2026-08-01)

> Tout ce qui précède comparait LLVQ au **FP16**. C'est la mauvaise référence :
> personne ne déploie du FP16 en local. La vraie question est *contre du 4 bits*,
> et elle n'avait jamais été posée. Voici la réponse, mesurée sur la même
> machine, le même modèle, le même jour.

## Le protocole

Qwen3-4B, M3 Max. Le 4 bits est produit **localement** depuis le checkpoint
FP16 déjà en cache — aucun téléchargement, aucun modèle tiers :

```bash
mlx_lm.convert --hf-path Qwen/Qwen3-4B -q --q-bits 4 --q-group-size 64 \
  --mlx-path qwen3-4b-mlx-q4
mlx_lm.generate --model qwen3-4b-mlx-q4 --prompt "…" --max-tokens 256 --temp 0
```

MLX plutôt que llama.cpp pour deux raisons : c'est le chemin natif Metal
d'Apple, donc l'adversaire le plus juste pour un noyau Metal ; et la conversion
GGUF aurait exigé un script Python absent de l'installation Homebrew.

## Le verdict

> 🚨 **Trois lignes de ce tableau ont été mesurées depuis, et elles étaient
> fausses — toutes dans le sens qui nous flattait.** Le tableau corrigé est
> celui du [`README.md`](../README.md#against-4-bit) ; la version ci-dessous est
> conservée pour la généalogie.
>
> | ligne | ce qui était écrit | mesuré le 2026-08-03/04 |
> |---|---|---|
> | RAM | 3,28 Go *(calculé)* | **9,79 Go de pic RSS en CPU, 17,41 Go en Metal** — 3,28 décrivait `Slot32`, que `bin/run` ne charge jamais |
> | débit | ~78,5 tok/s *(projeté)*, ×1,65 contre nous | **2,2 à 7,6 tok/s mesurés**, soit **×17 à ×58** contre nous : `bin/run` n'a pas de cache KV |
> | qualité du q4 | « ~1-2 % de dégradation » | **jamais mesurée**, ni ici ni ailleurs. La case est **vide, pas faible** |
>
> Les 2,39 Go et les 129,8 tok/s du bras MLX n'ont eux non plus **aucune trace**
> conservée (le 2,39 est un pic d'allocateur MLX, pas un RSS). Seule la ligne
> **disque** est mesurée des deux côtés.

| | MLX 4 bits | LLVQ 2 bits | |
|---|---|---|---|
| **disque** | 2,263 Go | **1,771 Go** | **×1,28** pour nous |
| **RAM** (mesuré / calculé) | **2,39 Go** | 3,28 Go | ×1,37 **contre** nous |
| **débit** (bout en bout, mesuré) | **129,8 tok/s** | ~78,5 tok/s *(projeté)* | ×1,65 **contre** nous |
| **qualité** | ~1-2 % de dégradation | **×1,386** | franchement contre nous |
| **bits/poids effectifs** | 4,50 | 3,52 disque / **5,51 RAM** | ⚠️ voir ci-dessous |

> ⚠️ **La dernière ligne ne se lit pas comme un rapport : « 5,51 contre 4,50 »
> mélange deux comptabilités**, et le dossier l'a déjà corrigé une fois
> (`docs/cheatsheet-defense.md` § « Les chiffres à connaître par cœur »,
> `docs/portage-noyau-cuda.md` §0.3). Le 5,51 est la comptabilité `thesis` des
> **projections seules** (payload + bases + queue f32 + échelles de ligne f32) ;
> le 4,50 est le q4 sur **tous** ses poids, embedding quantifié compris. À
> convention identique **poids seuls**, `docs/fiche-4b.md` §5.3 donne
> **6,5245 contre 4,5006, soit ×1,45 contre nous**. Le 5,51 reste juste pour
> **décrire notre format** — il n'est pas un terme de comparaison. *(La
> correction va dans le sens de cette section : à convention homogène l'écart
> est plus large, pas plus étroit.)*

**Sur un 4B, le 4 bits nous domine sur tous les axes sauf le disque, et de peu.**
Les 129,8 tok/s sont stables à 0,5 % près sur trois runs, mesurés de bout en
bout — attention, normes et cache KV compris. Nos 78,5 sont une *projection*
qui exclut tout ça : l'écart réel est au moins ×1,65, probablement pire.

## La leçon, et elle est structurelle

Le gain de place de LLVQ est **sur le disque** (3,52 b/poids). Mais le format
que le noyau rapide lit en RAM coûte **5,51 b/poids** dans la comptabilité
`thesis` — et à convention homogène poids seuls, **6,5245 contre 4,5006 au q4,
×1,45 contre nous** (`docs/fiche-4b.md` §5.3). Dans les deux comptabilités le
sens est le même, et c'est ce qui compte ici : la forme rapide est **plus
grosse** que du 4 bits. La vitesse a été achetée avec les bits mêmes qui
justifiaient le 2 bits.

C'est visible en extrapolant à 70B, là où la thèse est censée vivre :

| 70B | taille | tient sur… |
|---|---|---|
| FP16 | 140 Go | rien de local |
| **MLX 4 bits** | **39,4 Go** | Mac 48 Go |
| LLVQ `Slot32` (rapide) | **48,2 Go** | ❌ *pire que le 4 bits* |
| LLVQ `Grouped32` (lent) | 29,3 Go | Mac 32 Go ✅ |
| LLVQ sur disque | 19,0 Go | — |

**Le format qui va vite ne rentre pas mieux que du 4 bits ; le format qui rentre
mieux ne va pas vite.** On n'a pas encore les deux à la fois, et c'est *le*
problème à résoudre — pas un détail d'optimisation.

## Ce que ça ne dit pas

- **Le noyau reste une contribution réelle.** Un décodeur Leech multi-coquilles
  fusé qui bat le FP16 de 2,07× n'existe nulle part ailleurs, le papier compris
  (mono-coquille, plus lent que QTIP). Ce qui est réfuté, c'est le *produit* sur
  un 4B, pas l'ingénierie.
- **La qualité n'est pas mesurée sur des tâches.** ×1,386 de perplexité contre
  ~1-2 % pour le 4 bits est un écart massif, mais aucun MMLU n'a été passé, ni
  chez nous ni sur le 4 bits de cette machine.
- **Le régime batché n'est pas testé** — et c'est celui d'un cloud.

## Les trois sorties possibles

1. **Fermer l'écart de RAM.** Quantifier le `lm_head` (389 M poids encore en
   f16, 0,778 Go) descend `Slot32` à 2,77 Go et `Grouped32` à 1,68 Go — ce
   dernier passe *sous* MLX. Le levier était identifié depuis juillet ; il
   devient prioritaire.
2. **Rendre `Grouped32` rapide.** C'est le vrai sujet : 3,35 b/poids à vitesse
   utile battrait le 4 bits sur la place *et* tiendrait la route en débit. Le
   passage `Flat32` → `Slot32` a montré qu'un changement de format bien choisi
   vaut 2,4× ; il reste peut-être une forme intermédiaire.
   🏷️ Le 3,35 est en comptabilité `rtbits` (payload + bases) ; le même layout
   pèse **3,498** en métrique thesis, et `bin/thesis` le mesure à **0,69×
   [0,69–0,69]** le FP16 sur les 252 projections du modèle entier
   (`docs/mesures/k1-metal-2026-08-05.txt`, 2026-08-05). Ne pas aligner les deux
   comptabilités.
   ⚠️ **Le rapport est formé round par round**, puis résumé en médiane et plage
   sur les 5 rounds gardés (7 rounds, 2 jetés, tous les bras dispatchés à chaque
   round dans le même ordre). Ce **n'est pas** le quotient des colonnes « min
   ms » du journal : diviser un minimum par un minimum mêlerait deux rounds qui
   n'ont jamais coexisté. Le 3,498 b/poids, lui, est exact et se reproduit au
   chiffre ; les millisecondes dérivent d'un run à l'autre.
3. **Assumer le créneau.** Le 2 bits ne sert que là où le 4 bits **ne rentre
   pas** : 70B sur 32 Go, 405B sur 128 Go. Sur ces points-là, `Grouped32` gagne
   même lent, parce que l'alternative est *ne pas charger le modèle*.

Aucune de ces trois n'est acquise. La comparaison a coûté une heure et vaut
plus que la journée de noyau qui l'a précédée.

## La sortie n° 2, mesurée : plafonner les niveaux (`bin/lcap`, puis `bin/rtbits`)

La largeur du payload runtime est `34 + 24(L−1)` bits, où L est le nombre de
magnitudes distinctes du bloc — **zéro compris**. Donc L *est* le coût mémoire,
et le plafonner est le seul bouton qui échange directement distorsion contre
RAM. Mesuré sur source gaussienne, 20 000 blocs d'entraînement + 20 000
d'évaluation, shape–gain 1 bit, centroïdes ajustés sur le train :

| L max | codebook | index | b/dim | MSE | rétention | RAM b/poids *(compta `lcap`)* |
|---|---|---|---|---|---|---|
| 5 *(actuel)* | 2,81·10¹⁴ | 48 b | 2,0417 | 0,0725 | 92,72 % | 5,667 |
| **4** | 2,61·10¹⁴ *(93 %)* | 48 b | 2,0417 | 0,0730 | **92,46 %** | **4,667** |
| **3** | 6,23·10¹³ *(22 %)* | 46 b | 1,9583 | 0,0870 | **89,94 %** | **3,667** |
| 2 | 7,53·10¹⁰ | 37 b | 1,5833 | 0,1583 | 83,97 % | 2,667 |

> ⚠️ **Table révisée le 2026-08-01 (§A5).** Le banc ajustait et arrondissait le
> gain sur la **projection** `⟨x,v̂⟩` ; `LeechShapeGain` l'ajuste et l'arrondit
> sur la **norme** du bloc. Toutes les rétentions perdent 0,3 à 1,7 point.
> **Aucune décision ne change** : les écarts entre plafonds bougent à peine, et
> `L ≤ 3` reste sous les 4,50 b/poids du 4 bits. Ce qui change, c'est que la
> table décrit enfin le quantifieur qui a produit l'artefact.

> 🚨 **La colonne « RAM b/poids » ci-dessus n'est PAS comparable aux colonnes de
> RAM des autres documents** (corrigé le 2026-08-05, jalon K-1(a)). C'est la
> comptabilité de `bin/lcap`, qui facture à **chaque bloc la largeur du
> plafond** au lieu de celle du max de son groupe, et qui **ne compte pas les
> bases**. Elle majore donc le format actuel et minore les plafonds. Elle reste
> ici parce que c'est elle qui a produit les colonnes MSE/rétention, qui sont
> bonnes ; ne la sortez pas de son tableau.

### Les mêmes chiffres, mesurés sur l'artefact réel (`bin/rtbits`, 2026-08-05)

Compté sur `~/llvq-q4b.llvq` — 981 Mo, projections seules du Qwen3-4B publié —
soit **4 708 800 groupes de 32 blocs, 150 681 600 blocs**. Comptabilité :
**payload + une base `u32` par groupe, stride arrondi à l'octet**, c'est-à-dire
ce qu'une lane lit réellement, rapporté aux poids quantifiés.
*(Corrigé le 2026-08-05, deuxième passe : la première rédaction écrivait
« 113 011 200 blocs », soit 4 708 800 × **24** — un nombre de groupes multiplié
par la taille d'un bloc au lieu de celle d'un groupe. 4 708 800 × 32 =
150 681 600, chiffre imprimé par `rtbits`, cf.
`docs/mesures/k1c-rtbits-2026-08-05.txt`.)*

| | RAM b/poids *(compta `rtbits`)* |
|---|---|
| `Slot32` aujourd'hui | **5,3756** |
| sous plafond `L ≤ 4` | **≤ 4,7083** |
| **gain** | **0,667 b/poids, soit 12,4 %** |

Distribution mesurée des max de niveaux par groupe :

| max du groupe | groupes | part | stride |
|---|---|---|---|
| L = 3 | **1** | — | — |
| L = 4 | 1 566 710 | 33,272 % | 14 o |
| L = 5 | 3 142 089 | 66,728 % | 17 o |

Moyenne des max par groupe : **4,667 niveaux** ; moyenne par bloc : **3,726**.

> ⚠️ **Coïncidence numérique à ne pas lire comme une égalité.** Les 4,667
> ci-dessus sont des **niveaux**, la ligne `L max = 4` du tableau `lcap` porte
> 4,667 **b/poids**. Deux quantités sans rapport qui tombent sur le même
> chiffre.

> **Le statut logique du 4,7083 est un majorant inconditionnel, pas une
> simulation.** `L ≤ 4` implique `width_slot ≤ 9 + 1 + 24·4 = 106 bits =
> 14 octets`, donc **tout** groupe a un stride ≤ 14 o. Et il est *atteint* dès
> qu'un groupe porte un bloc à 4 niveaux, parce qu'un bloc déjà à `L ≤ 4` garde
> sa classe sous le plafond — son mot de code est l'argmin sur la boule
> entière, il reste l'argmin sur un sous-ensemble qui le contient encore.
> Mesure : **4 708 799 groupes sur 4 708 800** portent un tel bloc. Ce n'est donc
> ni une hypothèse d'indépendance ni une probabilité : c'est un **compte**.

⚠️ **Aucune mesure `rtbits` n'existe pour `L ≤ 3` ni pour `L ≤ 2`.** Ces deux
lignes restent dans l'ancienne comptabilité `lcap` (3,667 et 2,667) et n'ont
**pas** été remesurées sur l'artefact. Tout ce qui s'appuie dessus plus bas est
à lire comme une indication, pas comme un chiffre.

> 🔎 **Recoupement indépendant.** `bin/matvec`, qui compte par un autre chemin de
> code et sur une seule couche (`gate_proj`), rend **5,375 b/poids**, soit les
> 5,3756 que `rtbits` obtient sur le modèle entier. Deux implémentations, un
> seul chiffre.

**`L ≤ 4` est quasi gratuit** : −0,26 point de rétention sur source gaussienne
pour **−0,667 b/poids** de RAM mesurés sur l'artefact. On jette 7 % du codebook
et on ne perd presque rien — la prédiction haute dimension se vérifie. À prendre
sans discuter. *(La lecture « −1 bit/poids » qui figurait ici était la différence
5,667 − 4,667 de la table `lcap`, qui facture le plafond à tous les blocs ; le
gain mesuré est 0,667.)*

**`L ≤ 3` est un vrai arbitrage** : −2,52 points de rétention sur source
gaussienne, pour −2 bits/poids **en comptabilité `lcap` uniquement**. Il ferait
passer le format sous le 4 bits (3,667 contre 4,50), et il baisse aussi le débit
disque (1,958 b/dim contre 2,042).
⚠️ Le 3,667 n'a **pas** d'équivalent mesuré : le plafond `L ≤ 3` n'a pas été
passé sur l'artefact, ni en comptabilité `rtbits` ni en métrique thesis. Et il
mord bien plus fort que `L ≤ 4` — **un seul groupe sur 4 708 800 a aujourd'hui un
max ≤ 3**, là où `L ≤ 4` laisse déjà 33,272 % des groupes inchangés.

Bénéfice de bord : sous plafond, **le stride de groupe devient uniforme** — plus
de table de bases, plus de padding par groupe, et un décodage plus court.
⚠️ Ce n'est *pas* parce que les blocs auraient tous la même largeur : sous
`L ≤ 4` un bloc a `L ∈ {1,2,3,4}`, donc **34, 58, 82 ou 106 bits — quatre
largeurs, pas une**. Ce qui devient uniforme, c'est le **stride**, parce que le
groupe est dimensionné sur son max et que la mesure ci-dessus montre que
4 708 799 groupes sur 4 708 800 portent déjà un bloc à 4 niveaux : sous plafond
ils prendraient donc tous 14 octets. La conclusion — plus de table de bases —
tient, mais elle repose sur la statistique des groupes, pas sur la largeur des
blocs. Le noyau devrait être *plus* rapide, pas moins.

### Ce que ça change à 70B

> 🚨 **Ce tableau mélange deux comptabilités, et c'est la faute exacte que le
> jalon K-1(a) existait pour supprimer** (signalé le 2026-08-05).
> `Slot32 actuel = 48,2 Go` est en métrique **thesis** (5,51 b/poids : payload +
> bases + queue f32 + échelles de ligne f32) ; `L ≤ 4 = 40,8 Go` et
> `L ≤ 3 = 32,1 Go` sont en métrique **`lcap`** (4,667 et 3,667 : plafond
> facturé à tous les blocs, bases non comptées). Les lignes **ne sont pas
> commensurables entre elles** et le tableau n'a pas été recalculé, faute des
> chiffres pour le faire proprement.

| en RAM | taille | comptabilité | tient sur |
|---|---|---|---|
| FP16 | 140 Go | — | rien |
| `Slot32` actuel | 48,2 Go | **thesis** (5,51) | Mac 64 Go |
| `L ≤ 4` | 40,8 Go | **`lcap`** (4,667) | Mac 64 Go |
| MLX 4 bits | 39,4 Go | 4,50 b/poids | Mac 48 Go |
| `L ≤ 3` | 32,1 Go | **`lcap`** (3,667) | Mac 48 Go |

Le seul report que ces mesures permettent, la queue et les échelles de ligne ne
bougeant pas sous le plafond : **en métrique thesis,
`L ≤ 4` vaut `5,510 − 0,667 = 4,843 b/poids`.** Soit **au-dessus** des 4,50 du
4 bits, et non en dessous. La conversion en Go à 70B n'est pas refaite ici : ce
serait dériver un chiffre qui n'a pas été mesuré.

⚠️ **La conclusion « `L ≤ 3` reprend l'avantage mémoire, donc petit ET rapide »
n'est plus soutenue par ces chiffres.** Elle reposait sur le 3,667 de `lcap`,
comptabilité qui ne compte pas les bases et qui n'a pas été passée sur
l'artefact pour ce plafond. Ce qui est mesuré, c'est `L ≤ 4` : 4,7083 b/poids en
comptabilité `rtbits`, 4,843 en métrique thesis — dans les deux cas **au-dessus
du 4 bits**. La question « petit ET rapide » reste **ouverte**.

⚠️ Source gaussienne, une seule graine, pas de GPTQ. Les vrais poids ne sont
pas gaussiens et la boucle GPTQ déforme leur distribution : ceci **indique**,
ça ne tranche pas. L'A/B sur 3 blocs du vrai modèle tranche — et vu ce que
MMLU vient de révéler (on perd déjà plus de capacités que le papier), la
mesure à surveiller n'est pas la perplexité mais MMLU.
