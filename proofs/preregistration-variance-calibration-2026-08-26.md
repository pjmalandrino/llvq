# Pré-enregistrement — la variance de calibration : où est le seuil de résolubilité

**Écrit et commité AVANT le premier run.** À tamponner (`ots stamp`) avant la
première milliseconde de mesure, selon la règle du §7 de `CLAUDE.md`.

Ce document couvre **la grille 0.6B et ses contrôles**. Le transfert au 4B (§5,
étage 4) part sur **go séparé**, après un gate écrit ici.

Il ne dépend d'aucun autre pré-enregistrement et n'en amende aucun. En
particulier il **n'amende pas**
[`preregistration-volume-calibration-2026-08-25.md`](preregistration-volume-calibration-2026-08-25.md)
(sha256 `33fd4932…`, tamponné) : celui-ci mesure un **effet moyen** de volume au
4B, celui-là mesure une **variance**. Deux estimands, deux documents.

---

## §1 — La question, et le mécanisme qui se chiffre

GPTQ ne se contente pas d'arrondir : il **compense**. À chaque bloc de colonnes
quantifié, il retouche les colonnes suivantes pour annuler une part du dégât sur
la sortie de la couche. Le choix de cette retouche est dicté par la hessienne
`H = AᵀA/N`, estimée en faisant passer un échantillon de texte dans le modèle.

**Donc le texte de calibration entre dans les poids produits.** Change le texte,
tu changes `H`, tu changes la correction, tu changes le modèle.

Le problème d'estimation se chiffre. Sur la couche la plus large du 4B,
`down_proj`, `H` fait 9 728 × 9 728 et on l'estime sur 131 072 tokens — **13,5
exemples par dimension**. À `q = n/N = 0,0742`, Marchenko–Pastur étale les
valeurs propres empiriques d'une population **blanche** sur `[0,53 ; 1,62]` fois
la vraie (*calculé*) : un facteur **3,1 de pure erreur d'échantillonnage**, quand
l'amortissement `λ·mean(diag H)` à `λ = 1e-2` n'ajoute que **1 %** de la valeur
propre moyenne.

**Deux mesures du dépôt disent que ce n'est pas théorique.**

| journal | ce qu'il mesure |
|---|---|
| [`f5-graines-4b-2026-08-19.txt`](../docs/mesures/f5-graines-4b-2026-08-19.txt) | trois runs complets du 4B, `LLVQ_CALIB_SEED ∈ {1,2,3}` : ppl **16,7425 / 15,8836 / 15,1027**, étendue 10,3 %, **σ = 5,2 %** (n = 3), les trois paires appariées RÉSOLUES |
| [`gain-ab-gate-0.6b-2026-08-25.txt`](../docs/mesures/gain-ab-gate-0.6b-2026-08-25.txt) | quatre bras à débit constant, deux tirages : **le classement s'inverse**, étendue intra-bras **13,9 %** > étendue inter-bras **10,6 %** |

L'inversion de rang est le fait porteur. σ dit « il y a du bruit » ; l'inversion
dit **« la conclusion bascule »**.

**Ce que ce document cherche n'est donc pas « la calibration compte »** — c'est
connu, et l'ablation de volume est un genre déjà occupé. C'est **où passe le
seuil de résolubilité**, en fonction de quatre facteurs : le volume de
calibration, le débit de bits, la **famille de quantifieur**, et la largeur du
modèle.

🚨 **Réserve écrite avant la première mesure, et elle peut tout désamorcer.**
Nous calibrons sur 131 k tokens ; le papier amont, QuIP# et QTIP utilisent
~6 100 séquences de DCLM-edu ≈ **12,5 M tokens, 96× plus**. Si σ suit `1/√N`,
×96 divise σ par ~9,8 : de 5,2 % à ~0,5 %. **Le sable mouvant est peut-être le
nôtre seul.** Ce document existe pour trancher ça, pas pour le supposer. Repère
utile au passage : le défaut d'AutoGPTQ que les praticiens lancent réellement est
**128 échantillons de 2048 = 262 k tokens**, soit ×2 de notre point et 1/48ᵉ de
celui des papiers.

---

## §2 — Les quatre bras, et l'unique variable

**Une seule chose bouge entre deux runs d'une même cellule : `LLVQ_CALIB_SEED`.**
Tout le reste — modèle, corpus, rotation, damping, dtype, blocs, threads — est
identique et figé ci-dessous.

| bras | codebook | b/poids | rôle |
|---|---|---|---|
| `leech1c12` | shape–gain, 1 bit de gain, boule ≤ 12 | **2,1656** *(mesuré, journal du gate)* | VQ à codebook **intabulable** (1,1·10¹⁴ points) |
| `scalar-g128-b3` | INT3 groupwise, groupe 128, échelle+zéro asymétriques | 3 + 32/128 = **3,25** *(calculé)* | le standard du domaine, bras **porteur** de la courbe |
| `scalar-g128-b2` | idem, 2 bits | 2 + 32/128 = **2,25** *(calculé)* | point de **stress** — cf. la réserve ci-dessous |
| `scalar-g128-b4` | idem, 4 bits | 4 + 32/128 = **4,25** *(calculé)* | l'ancre haute de l'axe bits |

**Pourquoi le scalaire est obligatoire.** Si la variance n'est montrée que sur
Λ₂₄, c'est une curiosité sur une méthode dont le §0 du backlog établit qu'elle
n'a pas d'avenir produit. Sur un quantifieur scalaire groupwise — le défaut
d'AutoGPTQ, `group_size=128`, `static_groups=False` — le résultat devient une
propriété de **GPTQ**, pas de notre codebook. C'est ce qui décide si ce travail
est une note de bas de page ou un résultat de domaine.

⚠️ **Réserve déclarée d'avance sur `scalar-g128-b2`.** Le 2 bits scalaire nu est
connu pour être catastrophique — c'est pourquoi personne ne le déploie sans
autre chose. Sa perplexité peut être si dégradée que σ y mesure la **pathologie**
et non la calibration. Ce bras est un point de stress ; **l'axe bits se lit sur
b3 et b4**, et une σ aberrante à b2 ne réfute rien.

⚠️ **Le débit n'est pas apparié entre familles** (2,1656 contre 3,25). Comparer
σ entre `leech1c12` et `scalar-g128-b3` compare **deux points de qualité
différents**. C'est assumé : l'estimand est la **sensibilité relative** au
tirage, pas la qualité. Toutes les comparaisons inter-familles se font en σ
**relative** (en % de la ppl médiane de la cellule), jamais en ppl absolue.

**Corpus : C4, rôle `Calibration` (shard `00001`), et pas WikiText-2 train.**
Motif *calculé* et non préférentiel : WikiText-2 train porte ~2,09 M tokens, soit
**~1 020 fenêtres de 2048 = ×15,9 au maximum** — le corpus lui-même plafonnerait
la grille avant le dernier barreau. C4 n'a pas cette borne. L'évaluation reste
WikiText-2 **test**, disjoint par construction (`corpus.rs:187-193`).

---

## §3 — La grille, et les commandes verbatim

**Facteurs.** Volume `V ∈ {×1, ×2, ×4, ×8, ×16, ×32}` (×1 = 64 fenêtres × 2048 =
131 072 tokens) · bras (§2) · graine `s ∈ {1,2,3,4}`, **les mêmes quatre pour
tous les bras** — l'appariement est ce qui rend σ_diff calculable.

**Plan réduit, et il est réduit exprès** : la courbe de volume a besoin de
beaucoup de niveaux et de peu de bras ; l'axe bits de l'inverse.

| bras | volumes | graines | runs |
|---|---|---|---|
| `leech1c12` | les 6 | 4 | 24 |
| `scalar-g128-b3` | les 6 | 4 | 24 |
| `scalar-g128-b2` | ×1, ×8 | 4 | 8 |
| `scalar-g128-b4` | ×1, ×8 | 4 | 8 |
| | | **total** | **64** |

Commande, `<W>` = 64 · 128 · 256 · 512 · 1024 · 2048 et `<C>` le bras :

```
export LLVQ_MODEL=Qwen/Qwen3-0.6B LLVQ_CALIB=c4 LLVQ_THREADS=12
export LLVQ_CALIB_SEED=<s>
target/release/smoke <W> 2048 73 2048 metal nogs <C> 999 rot 2>&1 | tee $D/run.txt
```

`999` est la sentinelle « tous les blocs » (28 au 0.6B) ; `rot` est la rotation
d'entrée, comme partout ailleurs dans le dossier.

🚨 **`73` et non `12`.** L'évaluation porte sur les **73 fenêtres** du split
WikiText-2 test, pas les 12 historiques : facteur **2,47** sur toutes les barres
appariées pour quelques minutes de Metal. Les NLL par fenêtre sont sur **stderr**
à 9 décimales — le `2>&1 | tee` n'est pas cosmétique, sans lui l'appariement est
perdu.

---

## §4 — Le modèle de coût, mesuré et non estimé

Profil par phase d'un run 0.6B / 28 blocs / ×1, lu dans
[`gain-ab-2026-08-25-brut/gain-ab-1c12.log:49-55`](../docs/mesures/gain-ab-2026-08-25-brut/gain-ab-1c12.log) :
quantification **1347,0 s** (84,1 %) · factorisation **151,2 s** (9,4 %) ·
capture passe 1 **100,5 s** (6,3 %) · transfert f64 **2,6 s** · advance passe 2
**0,3 s** · écriture **0,0 s**. Segment total **1602 s**.

D'où deux termes, *mesurés* :

- **proportionnel au volume** = capture + advance ≈ **101 s par unité de ×1**
- **indépendant du volume, bras Leech** = 1347,0 + 151,2 + 2,6 ≈ **1501 s**
- **indépendant du volume, bras scalaire** ≈ **160 s** *(estimé)* — la recherche
  de plus proche voisin sur le réseau disparaît ; il reste la factorisation et un
  arrondi. ⚠️ **C'est le seul terme non mesuré du modèle, et l'étage 0 le pin.**

| V | fenêtres | Leech | scalaire | mémoire `hidden` f32 |
|---|---|---|---|---|
| ×1 | 64 | 26,7 min | 4,4 min | 0,54 Go |
| ×2 | 128 | 28,4 | 6,1 | 1,07 Go |
| ×4 | 256 | 31,7 | 9,4 | 2,15 Go |
| ×8 | 512 | 38,5 | 16,2 | 4,29 Go |
| ×16 | 1 024 | 51,9 | 29,6 | 8,59 Go |
| ×32 | 2 048 | 78,8 | 56,5 | **17,18 Go** |

**Total de la grille : 28,0 h de Mac, 0 $**, plus ~4 h d'évaluation à 73
fenêtres. Cinq nuits.

🚨 **Deux plafonds de l'outillage, à lever AVANT le premier run — sinon les deux
derniers barreaux sont silencieusement faux.**

1. **`smoke.rs:876` demande `c4_calibration(8_000_000)` et `smoke.rs:897` clampe
   en silence** (`n_calib.min(train_ids.len()/calib_len)`). Le lot B a mesuré ce
   que 8 M caractères rendent : **847 fenêtres** — soit 4,61 caractères par token
   (*calculé*). Donc **×16 (9,67 M car.) et ×32 (19,3 M car.) retomberaient tous
   deux sur ×13,2**. Le correctif est au §Lot 0 du plan de travaux, et il doit
   **échouer franchement** plutôt que clamper, per la règle du §7 de `CLAUDE.md`.
2. **`smoke.rs:909-913` garde toutes les fenêtres résidentes** sur la carte. La
   colonne mémoire ci-dessus est *calculée* (`W × 2048 × 1024 × 4 o`) : ×32 tient
   dans 17,18 Go, donc la grille passe en f32 sur une machine à 69 Go. Au-delà de
   ×32, il faudrait `LLVQ_DTYPE=f16` — **hors périmètre de ce document.**

---

## §5 — Les étages, et les gates qui les séparent

### Étage 0 — le gate à 0 $ (≈ 1 h)

Trois runs, et **rien d'autre ne part avant qu'ils soient verts** :

| run | ce qu'il pin |
|---|---|
| `leech1c12`, ×1, graine 1 — **deux fois** | le **déterminisme** du pipeline |
| `scalar-g128-b3`, ×1, graine 1 | le terme fixe du bras scalaire (§4) |

🚨 **Le contrôle de déterminisme est la fondation de tout le document.** Si deux
runs à graine identique ne rendent pas la **même perplexité au dernier chiffre
imprimé**, alors σ mélange deux sources — la calibration et le non-déterminisme
du backend — et **aucune cellule de la grille n'est interprétable**. Dans ce cas
la grille ne part pas ; l'étage 0 devient une enquête sur le déterminisme.

**Gate A**, posé ici : le pipeline est déclaré déterministe si les deux runs
rendent une ppl identique à **1e-9 relatif** et une empreinte de tokens
identique. Sinon → arrêt, et le protocole est à repenser.

### Étage 1 — la grille 0.6B (≈ 28 h + 4 h d'évaluation, 0 $)

Les 64 runs du §3, dans l'ordre `leech` puis `scalar-b3` puis les ancres, chaque
bras balayant ses graines avant de changer de volume.

### Étage 2 — le contrôle d'invariance de backend (≈ 4 $)

**Deux cellules répliquées sur CUDA** (`l40sx1`, ~1,77 $/h *mesuré* sur les jobs
`g3-planes12x-servi` 0,79 $/27 min et `f2-p3-banc-qtip` 0,89 $/30 min) :
`leech1c12` ×1 et `scalar-g128-b3` ×1, k = 3 graines chacune, soit **6 runs**.

⚠️ **Motif écrit d'avance, parce qu'il est contre-intuitif : ce n'est pas pour
aller plus vite.** 84 % du run est l'encodeur, qui est **CPU** ; louer une carte
n'accélère pas le terme dominant. Le contrôle sert uniquement à établir que σ
n'est pas une propriété de Metal. Porter la grille entière sur CUDA coûterait
~48 $ *(calculé : 28 h × 1,77 $/h)* pour un résultat que Metal rend à 0 $.

**Ce que le contrôle exige** : le σ relatif des deux backends doit se recouvrir.
S'il ne se recouvre pas, la grille se publie **étiquetée Metal**, et l'écart
devient un résultat à part.

### Gate B — avant de dépenser un dollar au 4B

Posé ici, et il est la demande explicite de l'opérateur du 2026-08-26 :
*« avant de lancer le transfert vers le 4B pour les 70 $ on pourra tester qu'on
a un truc clean sur les premiers résultats. »* Les quatre conditions :

1. **Gate A vert** (déterminisme établi).
2. **σ_diff mesurable et non dégénéré** : à ×1, l'écart-type des différences
   appariées `leech − scalar-b3` sur les 4 graines est fini et non nul.
3. **La courbe σ(V) a une forme** : l'ajustement `log σ = a + β·log V` sur les 6
   volumes du bras `leech1c12` rend un `β` dont l'IC95 **exclut 0**.
4. **Aucun contrôle du §6 tombé** sur plus d'une cellule.

Si les quatre passent → étage 3 sur go budget. Sinon → le document se clôt sur
la grille 0.6B, qui reste publiable telle quelle.

### Étage 3 — le transfert au 4B (~70 $, go séparé)

| cellule | runs | coût *(estimé, base : préreg volume §3 et jobs.csv)* |
|---|---|---|
| `leech1c12` ×1, k=3 | **0** | **déjà payé** — les trois graines F5 |
| `leech1c12` ×8, k=3 | 3 | ~24 $ |
| `scalar-g128-b3` ×1, k=3 | 3 | ~21 $ |
| `scalar-g128-b3` ×8, k=3 | 3 | ~24 $ |
| | **9** | **~69 $** + ~2 $ d'évaluation |

Le 4B est la seule taille où la métrique de **capacités** existe : MMLU est au
hasard sur un 0.6B. Les trois artefacts F5 sont au bucket
(`f5-graines-2026-08-19/seed{1,2,3}/`), donc leur MMLU coûte ~0,5 $ pièce et non
un requantification.

**Le 8B est hors de ce document.** Repère : 11,48 $ par run à ×1 (*mesuré*, run
du 2026-08-02), donc ~140 $ pour une pente à deux volumes. À rouvrir seulement si
la pente 0.6B → 4B est ambiguë.

---

## §6 — Les contrôles, et si l'un tombe la cellule n'est pas publiée

| # | contrôle | comment il se lit |
|---|---|---|
| **C1** | **déterminisme** | deux runs à graine identique → ppl identique à 1e-9. Vérifié à l'étage 0, **re-vérifié une fois** en fin de grille. |
| **C2** | **volume réellement lu** | `smoke` imprime `N windows of L = T tokens`. `T` doit **égaler** le volume demandé. Sinon la cellule se publie à son volume **réel**, avec son étiquette. |
| **C3** | **empreinte de tokens** | identique sur les 64 runs côté évaluation. Un bras qui n'imprime pas la même empreinte n'est pas comparable. |
| **C4** | **baseline invariante** | la ppl f32 de référence doit sortir identique à chaque run : elle ne dépend d'aucun facteur, donc si elle bouge, le harnais a bougé. ⚠️ **Sa valeur n'est pas 19,5038** — ce chiffre est celui de **12** fenêtres à ctx 2048 (journal du gate du 2026-08-25). À 73 fenêtres la référence est **autre**, et elle se fige à la première mesure de l'étage 0. |
| **C5** | **débit effectif** | `bits/weight` imprimé par run, constant à l'intérieur d'un bras sur tous les volumes et toutes les graines. |
| **C6** | **mode d'échantillonnage** | **toutes** les cellules en `LLVQ_CALIB_SEED` (offsets tirés), **aucune** en préfixe contigu. Le confondant du 2026-08-25 — `CALIB_SEED` change le tirage *et* le mode — est ainsi éliminé par construction plutôt que déclaré. |
| **C7** | **backend** | §5 étage 2. |
| **C8** | **un seul chemin de lecture** | la perplexité de **tous** les bras se lit sur la boucle interne de `smoke`, jamais sur un fichier scellé. Contrainte imposée par le code : `quantize_model_capturing` refuse la capture pour tout codebook dont les codes ne décrivent pas la reconstruction, et `ScalarGroupwise` n'a pas de `BlockCode` — le bras scalaire **ne produit donc aucun artefact**. Lire le bras Leech sur son fichier scellé et le scalaire en mémoire ferait bouger deux choses à la fois : l'écart en magasin entre les deux chemins vaut 16,9617 (f32, en mémoire) contre 16,9415 (f16, scellé) au 4B, soit 0,12 % — petit, mais du même ordre que ce qu'on cherche à trancher. |

🚨 **C6 est la correction directe du seul confondant non levé du gate du
2026-08-25**, qui écrivait : *« on ne peut PAS séparer "autre tirage" de "autre
mode" »*. Ici il n'y a plus de bras en préfixe, donc plus rien à séparer.

---

## §7 — Le plan d'analyse, figé ici

**Estimands, dans l'ordre d'importance.**

1. **σ_diff** — écart-type, sur les graines, de la différence appariée entre deux
   bras à volume constant. C'est lui qui décide de la résolubilité : si l'effet
   de graine était purement additif, σ_diff serait nul et l'appariement
   sauverait toute comparaison.
2. **σ_niveau** — écart-type relatif de la ppl d'un bras à travers ses graines.
3. **β**, la pente de `log σ_niveau = a + β·log V`, ajustée **en poolant les
   cellules** plutôt que cellule par cellule. Motif : σ estimé sur k = 4 a 3 ddl
   et un IC très large ; c'est la pente qui est l'estimand, pas chaque σ.
4. **γ**, le coefficient de `bits` dans le même modèle, lu sur b3 et b4 (b2 exclu
   par la réserve du §2).
5. **δ**, le contraste de famille `leech − scalaire` sur σ relative.

**Inversion de rang.** Comptée et publiée : pour chaque paire de bras et chaque
volume, le signe de la différence est-il le même sur les 4 graines ? Le taux
d'inversion est un résultat à part entière, et il ne demande aucun modèle.

**Ce qui ne sera pas fait** : aucun test d'hypothèse sur σ lui-même à k = 4 (les
ddl ne le permettent pas honnêtement). Les IC sur `β`, `γ`, `δ` sortent d'un
bootstrap **par graine** (les graines sont l'unité de rééchantillonnage), 10 000
tirages, graine de bootstrap `0xb0075eed` — la même que `mmlupair`.

---

## §8 — Ce que ce document NE décide pas

- **Il ne mesure pas un effet moyen de volume.** C'est l'objet du
  pré-enregistrement du 2026-08-25, et les deux ne se soustraient pas.
- **Il ne dit rien de MMLU** : au 0.6B la métrique est au hasard. La capacité
  n'entre qu'à l'étage 3, et sur deux volumes seulement.
- **Il ne teste qu'une architecture.** `model.rs` est une passe avant écrite à la
  main pour Qwen3. Une seconde famille (Llama-3.2-1B) demanderait ~200-300 lignes
  plus un `oracle` contre `candle_transformers` — c'est **la limite de crédibilité
  n°1 de ce travail**, et elle est déclarée ici plutôt que découverte en revue.
- **Il ne compare pas à QTIP.** Nous avons son **noyau d'inférence**, pas son
  **quantifieur**, et le bras du banc tourne sur payload pseudo-aléatoire —
  interdit explicite de toute phrase de qualité. Un quantifieur treillis **à
  nous** (Viterbi sur un état 16 bits, ~200-300 lignes) serait le substitut
  propre ; il est hors périmètre.
- **Il n'autorise aucune dépense au-delà des ~4 $ de l'étage 2.** L'étage 3 part
  sur go séparé, après Gate B.

---

## §8bis — Écarts au protocole (journal, tenu à chaud)

*Vide à la signature. Tout écart constaté en cours d'exécution s'écrit ici, daté,
avec ce qu'il invalide — jamais par édition du corps de ce document, dont le
tampon atteste les octets.*

---

## §9 — Divulgation datée et prédictions signées

**Ce qui est connu à la signature** : σ = 5,2 % au 4B sur n = 3 (F5) ; l'inversion
de rang du 2026-08-25 sur n = 2 tirages ; le profil de coût du §4 ; les deux
plafonds d'outillage du §4 ; et le fait que le pipeline **n'a jamais été testé
pour son déterminisme**.

**Prédictions, écrites pour être opposables.** Le dossier a signé deux prédictions
fausses le 2026-08-25 ; celles-ci sont dans le même esprit.

| # | prédiction | ce qui la réfute |
|---|---|---|
| **P1** | Le pipeline est déterministe : deux runs à graine identique rendent la même ppl. | Un écart > 1e-9. **Si P1 est fausse, tout le reste tombe** — et c'est la prédiction que je tiens pour la plus sûre, donc la plus coûteuse à rater. |
| **P2** | σ décroît avec le volume, `β ∈ [−0,6 ; −0,4]` (loi en `1/√V`). | Un IC95 de β contenant 0, ou un β positif. |
| **P3** | Chez le scalaire, σ(3 bits) > σ(4 bits), rapport ≥ 1,5. | Un rapport < 1,2 ou de signe inverse. |
| **P4** | À volume égal, σ(`leech1c12`) > σ(`scalar-g128-b3`) — un codebook à beaucoup de degrés de liberté sur-exploite une hessienne bruitée. | σ(leech) ≤ σ(scalaire). **C'est la prédiction la plus intéressante et la moins fondée** : aucune mesure du dépôt ne la soutient, c'est un raisonnement. |
| **P5** | *(étage 3 seulement)* σ(4B) > σ(0.6B) à volume égal, parce que les exemples par dimension passent de 42,7 à 13,5. | σ(4B) ≤ σ(0.6B). |

🚨 **P5 est probablement fausse, et je l'écris quand même.** Les bribes existantes
pointent l'inverse : le 0.6B a montré des écarts de 5,9 à 13,9 % là où le 4B rend
σ = 5,2 %. Ce n'est pas comparable strictement — étendues contre σ, et le 0.6B
changeait aussi le mode d'échantillonnage — mais la tension est réelle. **C'est
elle qui rend l'axe d'échelle discriminant plutôt que décoratif**, et c'est la
seule raison de dépenser les 70 $ de l'étage 3.

**Ce qui invaliderait ce pré-enregistrement en entier** : que P1 tombe ; que C2
révèle que le volume demandé n'est jamais atteint au-delà de ×8 malgré le
correctif ; ou qu'une source de variance non déclarée soit trouvée dans le
pipeline avant la fin de la grille.
