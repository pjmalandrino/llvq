# Pré-enregistrement — P1 : ce qu'un décodage du rang coûte sur distribution réelle, et les seuils qui l'autorisent ou le ferment

**Date : 2026-08-13.** Écrit **avant toute mesure**. À cette heure :

- les deux décodeurs jugés ici — **cascade uniformisée** et **marche
  binomiale** — ont **zéro ligne de code**, sur GPU comme sur CPU ;
- `llvq-metal/src/bin/decode.rs` existe et a été exécuté **une fois**, le
  2026-07-31, sur des **codes synthétiques** dont tous les blocs portent
  4 magnitudes ;
- **ce run n'a pas de journal dans `docs/mesures/`.** Ses trois chiffres
  (sol 0,08 · masques 0,11 · rang-cascade 8,27 ns/bloc) ne survivent que dans
  la prose de [`docs/format-noyau.md`](../docs/format-noyau.md), qui les
  publie avec ses propres réserves (§ « Ce que ce chiffre ne couvre PAS »).
  **C'est une dette de provenance, et elle commande la règle §1.4 ci-dessous :
  aucun seuil ne se lit contre ces trois nombres.**

Le dernier verdict en date est celui du lot du 13
([`preregistration-2026-08-13.md`](preregistration-2026-08-13.md) et son
journal [`e1v-ordre-fichier-4b-2026-08-13.txt`](../docs/mesures/e1v-ordre-fichier-4b-2026-08-13.txt)) :
E3 mort en ordre-fichier à 2,9650 b/poids noyau contre 2,60, et `e1v-séparé`
vert **en largeur** à 2,3709 b/poids sous warp-scan — dont le décodeur n'existe
pas. Ce pré-enregistrement juge précisément ce décodeur-là.

> Il **hérite sans dérogation** des gardes du pré-enregistrement du
> [2026-08-10](preregistration-2026-08-10.md) (§7), de sa comptabilité (§6) et
> de sa règle de provenance — rien n'est recopié ici, c'est le même engagement.
>
> ⚠️ Ni signé GPG ni horodaté tant que l'opérateur ne l'a pas fait
> (`ots stamp proofs/preregistration-p1-2026-08-13.md`). **Cette fois le
> tampon est porteur** : il est demandé *avant* que la première ligne du banc
> soit écrite, contrairement au pré-enregistrement du lot du 13, dont
> l'antériorité ne repose que sur un mtime (cf. son commit `0cf05e1`).

---

## 0. Ce que ce banc peut et ne peut pas conclure

Il mesure un **décodage seul**, sur **Metal**, sur un **M3 Max**, en
**un bloc par lane**, sans matvec, sans réduction inter-lanes, sans tuilage.
Ce n'est pas le noyau servi et ça ne le sera jamais.

**Ce qu'un vert achète** : le droit d'être mesuré sur carte en P4, et rien
d'autre. **Ce qu'un rouge fait** : il ferme, et c'est là toute la rentabilité
du banc — un rouge à 0 $ épargne un job CUDA et une semaine de bijection.

🚨 **L'asymétrie s'arrête au seuil de 0,45 ns, qui est d'une autre nature que
les deux autres.** Les seuils de 1,5 et 2,0 ns sont des verdicts **sur la
mesure elle-même**, dans l'unité où elle est prise. Le 0,45 ns, lui, autorise
une **dépense CUDA** à partir d'un chiffre **Metal** : c'est une inférence
inter-matériel, et elle est plus faible que tout le reste de ce document. Elle
est posée d'avance pour ne pas être négociée après coup, pas parce qu'elle est
solide. Ce qui la fonde est écrit au §5, y compris ce qui ne la fonde pas.

## 1. La comptabilité, figée ici

**1.1 — L'unité est la nanoseconde par bloc**, et rien d'autre. Elle vaut
`(t_best − surcoût_de_soumission) / N`, avec `N = 2^24 = 16 777 216` blocs,
`t_best` = le meilleur de 15 répétitions après 3 chauffes, et le surcoût
mesuré par `Kernel::overhead(20)` dans le même processus. C'est la convention
déjà câblée dans `llvq-metal` (`lib.rs:263-296`) : **le minimum, pas la
moyenne** — un GPU partagé avec un compositeur a un plancher de bruit
au-dessus de lui, jamais en dessous.

**1.2 — `N = 2^24` est un minimum, pas un choix libre, et le surcoût n'y est
PAS du bruit.** À 2 M blocs le surcoût de soumission valait le travail mesuré ;
c'est l'un des trois défauts qui faisaient dire « 25 tok/s, c'est mort » avant
correction. Tout bras mesuré sur moins de `2^24` blocs est nul et non avenu.

🚨 **Mais `2^24` ne le rend pas négligeable : le dépôt le chiffre lui-même à
12 %** (« À 2 M blocs, le surcoût de soumission (~0,18 ms) valait le travail
mesuré. 16,7 M blocs le ramènent à **12 %** » — `docs/format-noyau.md:136-137`).
Recoupement : au bras `sol` mesuré à 0,084 ns/bloc, 2^24 blocs font 1,41 ms de
travail, et 0,18 ms de surcoût en font 12,8 % (calculé). Un bras qui rendrait
0,20 ns/bloc verrait donc **un huitième de son chiffre venir d'une
soustraction**. Conséquences, posées ici :

- l'`overhead` est mesuré **à chaque round**, pas une fois au début ;
- **sa dispersion est imprimée** (min, médiane, max sur les rounds gardés), au
  même titre que celle des bras ;
- si l'étendue du surcoût dépasse **la moitié** de l'écart entre deux bras que
  le verdict sépare, **ce verdict n'est pas rendu** : on ne tranche pas un
  écart plus petit que le bruit de la correction qui le produit.

**1.3 — Tous les bras dans un seul processus, un ordre de dispatch fixe,
tous dispatchés à chaque round.** Rapports formés **round par round**, jamais
comme quotient de deux minima issus de rounds n'ayant jamais coexisté (règle
de maison n°2). Un bras ajouté ne réordonne jamais les bras existants.

**1.4 — Les ancres se remesurent dans le même run, et les seuils se lisent
contre elles.** `sol` et `masques` sont redispatchés à chaque round. **Aucun
seuil, aucun rapport, aucune conclusion ne se lit contre un chiffre d'un autre
run.** Deux jeux historiques tombent sous cette règle, et il faut les nommer
tous les deux — la première version de ce document n'en nommait qu'un.

| run | chiffres | ce qu'ils valent ici |
|---|---|---|
| **2026-07-31**, `bin/decode` | sol 0,08 · masques 0,11 · rang-cascade 8,27 ns/bloc | contexte historique, rien de plus : **codes synthétiques à 4 magnitudes uniformes**, donc aucune divergence entre lanes |
| **2026-08-01**, `bin/decreal` | sol **0,084** · Fixed96 **0,152** (1,81× le sol) · Grouped32 **0,158** (1,89×) ns/bloc | **le vrai prior**, parce que ce run est déjà sur **blocs réels** du 4B publié — mais il ne fait pas seuil pour autant |

🚨 **Le second jeu est le plus dangereux des deux, et c'est celui que ce
document oubliait.** `decreal` a tourné le 2026-08-01 sur **16,7 M blocs réels
du 4B publié**, préfixes contigus des 252 matrices, chaque sortie vérifiée
(`docs/format-noyau.md:190-205`) : c'est la distribution même que P1 veut
mesurer, donc l'ancrage naturel — et l'ancrage naturel est exactement ce qui se
glisse dans un verdict sans qu'on s'en aperçoive. Or **il souffre de la même
dette de provenance que le jeu de juillet** : `grep -rln "decreal\|0,152\|0,158"
docs/mesures/` ne rend **rien**. Ces trois nombres ne vivent que dans la prose
de `format-noyau.md`, sans journal de run, sans machine nommée, sans
reproductibilité.

**Règle, donc :** les cinq bras de P1 se lisent **les uns contre les autres
dans le même run**, jamais contre 0,084 / 0,152 / 0,158 ni contre 0,08 / 0,11 /
8,27. Si le `sol` remesuré s'écarte notablement du 0,084 du 08-01, ce n'est
**pas** un résultat de P1 : c'est un signal qu'il faut expliquer avant de
publier quoi que ce soit (cf. §7).

**1.5 — La distribution est réelle, et le fichier est nommé.** Les blocs sont
tirés de l'artefact 4B scellé `~/llvq-q4b.llvq` (980 790 202 octets, celui
des sweeps du dépôt), donc avec ses proportions réelles de classes — 3 à 5
niveaux de magnitude, 4 pour 65,9 % des blocs, 50/50 pair/impair. Le banc du
07-31 mettait **4 magnitudes partout**, ce qui ne teste aucune divergence
entre lanes ; c'est la réserve principale que P1 existe pour lever.
⚠️ **Si le fichier est absent, le banc ÉCHOUE — il ne saute pas** (convention
de maison : un test qui saute quand son fichier manque doit échouer).

**1.6 — Le tirage est déterministe** : graine figée, imprimée en tête de
journal, et l'histogramme de classes du tirage imprimé à côté de celui du
fichier entier. Un tirage qui s'écarte de la distribution source invalide le
run avant tout chronométrage.

⚠️ **Et la couverture d'un tirage réel est bornée par le fichier, pas par le
codebook — à écrire dans le journal, jamais à sous-entendre.** Le 4B publié
porte **286 classes observées** et **zéro bloc origine**
(`docs/mesures/shell-distribution-4b-2026-08-10.txt:47`, produit par
`bin/classhist`). La table GPU en compte **384** (383 classes cap-13 +
l'origine, `llvq-metal/src/lib.rs:377-419`). Un tirage réel n'exerce donc
**ni la branche origine, ni les 82 classes de la coquille 13, ni les 15 classes
vides de la boule cap-12** — 98 entrées sur 384. Écrire « toutes les classes
sont couvertes » serait faux. Ces trois chemins se testent **séparément, sur
fixture synthétique**, comme le fait déjà `fixture_indices`
(`llvq-artifact/tests/e1c_format.rs:81-96`, qui ajoute explicitement l'origine
et les deux bornes de chaque classe) — et cette fixture fait partie de V0,
pas du banc.

## 2. Les bras, figés ici

Cinq, tous dans le même processus :

| bras | rôle | existe ? |
|---|---|---|
| `sol` | aucun décodage — lit les octets, accumule un produit scalaire | ✅ ancre |
| `masques` | masques imbriqués (`decode_payload`) — **l'ancre historique, PAS le chemin servi** ⚠️ | ✅ ancre |
| `cascade-archive` | l'unranking du format d'archive tel quel (récurrence u64 `M' = M·c/n`) | ✅ **étalon, pas décoration** — voir §4.3 |
| `cascade-uniformisée` | 24 itérations identiques pour toute classe, réciproques magiques par (classe, étage) en table, candidats en ILP, sélection branchless, **zéro indexation dynamique** | ❌ à écrire |
| `marche-binomiale` | le décodeur d'E1v : unranking par table `C(n≤24, k≤12)` tenue en L1, pas à compte fixe, **zéro division** | ❌ à écrire |

⚠️ **`masques` n'est le chemin servi d'aucune production, et la première
version de ce document l'écrivait.** Le layout servi est `Planes14`, sur CUDA
(`llvq-llm/src/fused.rs:68`). Côté Metal il n'existe pas :
`grep -rn "Planes" llvq-metal/src/` ne rend **rien**, et le MSL du dépôt ne
connaît que Fixed96, Grouped32, Flat32, Sorted32 et Slot32
(`llvq-metal/src/lib.rs:451-724`). Un journal qui dirait « on mesure la
production » sur ce bras mentirait. `masques` est ici pour une seule raison :
c'est le décodeur le plus rapide que la machine ait jamais exécuté, donc le
plancher pratique contre lequel un décodeur de rang se juge.

⚠️ **Le bras `cascade-archive` a une dette d'API à payer avant d'exister.** La
récurrence qu'il doit porter est écrite (`unrank_fast`,
`llvq-search/src/fastdec.rs:104`) mais elle est **privée** (`fn`, pas
`pub fn`), comme `struct FastClass` (:64-87) et le champ `classes` (:129) qui
portent la factorisation par classe dont elle a besoin. Deux issues, et le
choix se fait **avant** d'écrire : exporter (modification d'API sur un crate
`forbid(unsafe_code)` et sans dépendance), ou réécrire — et alors deux copies
de la même récurrence dans deux crates, c'est-à-dire précisément le
« malentendu partagé » que le §3.1 interdit. **La première issue est retenue** :
on exporte, et la référence reste unique.

Chaque bras **accumule un produit scalaire et écrit un float par bloc** — la
forme d'un noyau fusé. Aucun bras n'écrit les 24 poids décodés : c'est le
défaut n°1 du banc de 2026-07-31, qui mesurait ses propres écritures non
coalescées (« un banc mémoire déguisé en décodeur »). L'activation est chargée
une fois par threadgroup, jamais relue par itération (défaut n°2).

⚠️ **`Kernel::time` ne peut pas satisfaire le §1.3, et il ne faut pas
l'utiliser.** Il exécute warmup + reps d'un **seul** bras avant de rendre la
main (`llvq-metal/src/lib.rs:277-296`) : un banc écrit comme `decreal`
(trois `time()` successifs) produit des minima issus de rounds n'ayant jamais
coexisté. Le seul gabarit conforme du dépôt est la boucle manuelle de
`thesis.rs:871-901`. Le banc P1 écrit la sienne autour de `Kernel::dispatch`.
À savoir avant de l'écrire : **`Kernel::new` crée un `Device` ET une command
queue par bras** (`lib.rs:51`, `:61`), donc cinq bras entrelacés soumettent sur
**cinq files distinctes**, pas sur une file ordonnée. `thesis` vit avec et son
protocole tient ; c'est un fait à connaître, pas un obstacle.

⚠️ **Ne pas reprendre le pied de sortie de `decreal`.** Il imprime le repère
synthétique « 0,11 ns/bloc » (`decreal.rs:277-281`) que le §1.4 interdit
d'utiliser comme seuil, et il code en dur **400 Go/s** pour en dériver des
« plafond N tok/s » (`decreal.rs:274`) qu'un lecteur prendra pour des mesures —
défaut déjà relevé dans `docs/archive/plan-de-test-papier.md:194`. Le journal
de P1 n'imprime aucun tok/s dérivé d'une bande passante non mesurée.

## 3. V0 avant V1 — l'exactitude d'abord, sans exception

**Aucune milliseconde n'est chronométrée avant que le décodeur soit prouvé.**
Dans cet ordre :

1. **Référence CPU écrite indépendamment du shader**, comme
   `decode_masks_cpu` l'est aujourd'hui — pour qu'un malentendu partagé ne
   passe pas.
2. **Bloc à bloc**, sur un échantillon large couvrant toutes les classes du
   tirage, **plus la fixture synthétique du §1.6** pour l'origine et la
   coquille 13, qu'aucun tirage réel n'atteint. ⚠️ **L'étalon n'est pas le
   même pour les deux bras neufs** — voir le 🚨 ci-dessous.
3. **Sweep intégral des 150 681 600 blocs** du 4B scellé, harnais du sweep
   E1c (`llvq-artifact/tests/`), compte de blocs imprimé — pas un skip.

🚨 **La marche binomiale ne décode PAS l'ordre d'archive, et exiger d'elle une
égalité avec `FastDecoder::decode` était une impossibilité déguisée en
rigueur.** Le rang d'archive est un **rang de permutation de multiensemble**
(mixed-radix, lexicographique sur les suites de créneaux) ; une marche
binomiale unranke en **ordre combinatoire** sur des sous-ensembles. Passer de
l'un à l'autre *est* la ré-bijection CNS — c'est-à-dire **P5**, que ce banc est
censé conditionner. Un bras `marche-binomiale` vérifié contre `fd.decode`
exigerait donc le transcodeur qu'il gate : une dépendance circulaire, et un
V0 que le bras ne pourrait jamais franchir. Les deux étalons, fixés ici :

| bras | étalon de V0 | pourquoi |
|---|---|---|
| `cascade-archive`, `cascade-uniformisée` | **égalité point à point avec `FastDecoder::decode`** (`llvq-search/src/fastdec.rs:321`), sweep intégral compris | ces deux-là décodent **l'ordre d'archive** : même entrée, même sortie exigée |
| `marche-binomiale` | **aller-retour sur sa propre bijection** — `rank → arrangement → rank` sur **tous** les rangs de chaque classe assez petite, et sur un échantillon large des autres — **plus** la preuve de bijection : les arrangements produits sur `0..cardinalité` sont **deux à deux distincts** et leur compte égale `⌈cardinalité⌉` de la classe | elle définit **son** ordre ; ce qu'il faut prouver n'est pas qu'elle rend le même rang, c'est qu'elle est **une bijection sur le même ensemble** |

⚠️ **Conséquence sur ce que le bras `marche-binomiale` mesure, à écrire dans le
journal.** N'ayant pas de transcodeur, il est alimenté par des rangs **tirés
uniformément dans `0..cardinalité`** de la classe **réelle** de chaque bloc
réel du tirage §1.5. Le coût du décodage ne dépend des données que par la
**classe** (ses comptes, ses radices) et par la **magnitude du rang** : ces
deux-là sont donc exercés à leur distribution réelle. Ce qui n'est **pas**
exercé, c'est la corrélation éventuelle entre un bloc et son rang CNS
particulier — et rien n'indique qu'une telle corrélation existe, la marche
étant à compte fixe. **Le bras reste une mesure de vitesse honnête ; il n'est
pas une preuve que le format E1v décode le modèle publié.** Cette preuve-là est
`C2` de P5, et elle exige la ré-bijection.

🔎 **Le point 2 est un ajout, pas un acquis, et il faut le dire.** Aucun banc
Metal du dépôt ne vérifie aujourd'hui contre `FastDecoder::decode` : `decreal`
et `thesis` vérifient contre le **transcodeur** (`RuntimeBlocks::decode_block`),
ce que le dépôt documente lui-même
(`docs/archive/portage-noyau-cuda.md:513`). La chaîne existe — ce décodeur
d'exécution est épinglé bit pour bit sur `Indexer::decode`, 10 mutants tués
(`docs/format-noyau.md:192-193`) — mais elle est **transitive**. P1 exige la
comparaison **directe**, qui est plus forte et qui n'a pas de précédent côté
Metal.

📍 **Placement du code, contraint par le graphe de dépendances et fixé ici** :
référence CPU des décodeurs neufs dans **`llvq-search`** (visible du banc *et*
du sweep), shader dans `llvq-metal`, sweep dans **`llvq-artifact/tests/`** —
seul endroit d'où l'on peut ouvrir un `.llvq`, `llvq-search` n'ayant aucune
dev-dependency. Écrire la référence CPU dans un bin de `llvq-metal` rendrait le
sweep du point 3 impossible sans dupliquer le code, c'est-à-dire sans créer le
malentendu partagé que le point 1 interdit.

**Tout écart enterre le bras sans banc.** On ne chronomètre pas un décodeur
dont la reconstruction n'est pas prouvée (règle héritée du 08-11).

## 4. Les seuils, posés avant la première mesure

Les trois valeurs sont celles proposées par la contre-expertise du 2026-08-13
(passation §4, fiche P1), **figées ici sans amendement**.

### 4.1 Les verdicts par bras

| bras | vert | rouge |
|---|---|---|
| `marche-binomiale` (le décodeur d'E1v) | ≤ **1,5 ns/bloc** | > 1,5 ⇒ **mort** |
| `cascade-uniformisée` | ≤ **2,0 ns/bloc** | > 2,0 ⇒ **mort** |

### 4.2 Le gate CUDA, plus strict que les deux

**Le bras cascade/marche du job P4 n'est autorisé que si le meilleur des deux
décodeurs rend ≤ 0,45 ns/bloc.** Trois régimes, décidés d'avance :

| mesure Metal du meilleur bras | conséquence |
|---|---|
| ≤ **0,45 ns** | le bras CUDA de P4 est autorisé (il reste soumis au go de dépense) |
| entre **0,45 ns** et son seuil de kill | le bras **survit** comme point de la courbe et **n'achète aucun bras CUDA** — il faut une idée neuve, pas un job |
| au-dessus de son seuil de kill | **mort** |

### 4.3 Le kill aval, et l'étalon qui peut tout fermer

- **Si les deux bras sont rouges** : le **package C meurt** (le 70B de poche
  n'a pas de décodeur), et le **package B se réduit au prefill pur**, où le
  décodage s'amortit sur le nombre de tokens du lot.
- **L'étalon `cascade-archive` peut fermer la ligne par le haut, et le
  critère est un nombre déjà posé dans ce document** : si
  **`cascade-archive` rend ≤ 2,0 ns/bloc** — le seuil de la cascade
  uniformisée, §4.1 — alors **E1v est mort-né**. Le raisonnement tient en une
  ligne : l'archive **existe**, elle est **prouvée**, elle est **plus petite**
  (2,19 b/poids noyau contre 2,3709 pour E1v), et elle vient de franchir la
  barre qu'on impose aux décodeurs neufs. Un décodeur qu'on n'a pas à écrire,
  qui pèse moins et qui passe le même seuil ne laisse aucun argument à un
  décodeur qu'il faudrait écrire, prouver sur 150,7 M blocs et transcoder.
  Ce bras n'est pas un contrôle décoratif : c'est le seul du lot qui puisse
  rendre tout le reste inutile.
  > 🚨 **La première version de cette clause disait « si elle passe la
  > tolérance d'un cadre capacity-first » — et cette tolérance n'est chiffrée
  > NULLE PART** (trois occurrences de l'expression dans `docs/` et `proofs/`,
  > aucune avec un nombre). Après la mesure, qui voulait ouvrir P5 aurait dit
  > qu'elle ne passe pas, qui voulait le fermer aurait dit l'inverse, et rien
  > n'aurait départagé. C'était la porte de sortie la plus large du document.
  > Corrigée en É1 (§7bis) — sans inventer de nombre : celui-ci était déjà là.

## 5. La prédiction, et ce qui ne la fonde pas

**Aucune fourchette n'est prédite pour les deux décodeurs neufs**, et c'est
délibéré : ils n'existent pas, un compte d'instructions sur du code non écrit
ne serait pas une prédiction mais un vœu. Le seul repère disponible est
l'écart historique entre masques et rang d'archive — deux ordres de grandeur —
et il ne dit rien d'une cascade uniformisée qui n'a jamais tourné.

**Ce qui fonde le derating du 0,45 ns**, énoncé pour qu'on puisse le
contester :

- **un compte niveau source a déjà été faux d'un facteur 2 sur ce noyau** —
  c'est le précédent Golay70, où le hissage de la logique de coset promettait
  au compte d'instructions ce que la carte n'a pas rendu (1,77× mesuré contre
  une fourchette estimée 1,9–2,4×) ;
- les 2,52 T op/s du banc de 07-31 viennent d'une **chaîne dépendante**, donc
  tout budget « opérations par bloc » dérivé de ce banc est **~2× pessimiste**
  dans un sens et inutilisable dans l'autre.

🚨 **Ce qui NE le fonde pas, et il faut le dire** : le 0,45 ns n'est pas
dérivé d'un budget mémoire du noyau fusé CUDA. C'est une marge de sécurité de
facteur ~2 appliquée à un jugement d'ingénierie, prise sur une machine qui
n'est pas la cible. **Elle est conservatrice par construction** : elle peut
tuer un décodeur qui aurait passé sur carte. C'est le sens qu'on préfère —
un faux négatif coûte une idée, un faux positif coûte un job et une semaine
de bijection.

Si un bras rend **mieux que 0,20 ns/bloc**, chercher l'erreur avant d'en faire
un titre : bras dégradé, tirage non représentatif, boucle éliminée par le
compilateur faute d'être observable, ancres non reproduites.

## 6. Les issues, et ce que chacune fait au dossier

| issue mesurée | conséquence, décidée d'avance |
|---|---|
| marche binomiale ≤ 0,45 ns | **P5 s'ouvre** (re-bijection CNS) et le bras CUDA de P4 est autorisé ; le package C reste vivant |
| cascade uniformisée ≤ 0,45 ns **mais** marche binomiale > 0,45 | le bras CUDA de P4 est autorisé (§4.2, « le meilleur des deux ») et **P5 ne s'ouvre PAS** — les deux règles sont distinctes, et c'est le seul cas où elles divergent |
| marche binomiale ∈ ]0,45 ; 1,5] ns | E1v survit en largeur et **reste sans chemin d'exécution** ; ni P5 ni bras CUDA — la publication le dit comme point de courbe |
| marche binomiale > 1,5 ns | **E1v est mort comme format servi** ; sa largeur de 2,3709 reste un résultat de comptage, rien de plus |
| cascade uniformisée ≤ 2,0 ns | la famille cascade reste candidate pour le chemin d'archive |
| cascade uniformisée > 2,0 ns | la famille cascade se referme sur Metal |
| `cascade-archive` ≤ **2,0 ns/bloc** | **E1v mort-né** (§4.3) — l'archive existe, est prouvée, pèse moins (2,19 contre 2,3709) et vient de passer la barre des décodeurs neufs ; P5 ne s'ouvre pas, quel que soit le résultat de la marche binomiale |
| `cascade-archive` > **2,0 ns/bloc** | l'archive reste un décodeur de chargement, pas un décodeur servi ; les deux bras neufs gardent leur raison d'être et leurs seuils du §4.1 s'appliquent tels quels |
| les deux neufs rouges | package C mort, package B réduit au prefill pur (§4.3) |

**Aucune de ces issues ne bloque P2 ni P3**, qui sont dus quel que soit le
verdict et ne dépendent d'aucun décodeur.

## 7. Ce qui invaliderait ce pré-enregistrement

- **si les ancres `sol` et `masques` ne se reproduisent pas d'un round à
  l'autre** au sens de la règle de résolution du 08-10, le banc a dérivé et
  **aucun chiffre du run n'est publiable** avant d'en connaître la cause ;
- **si un bras échoue la vérification f64 / `FastDecoder`** (§3), le bras
  n'existe pas — pas de chronométrage, pas de verdict, correction d'abord ;
- **si l'histogramme du tirage s'écarte de celui du fichier source** (§1.6),
  la distribution réelle n'est pas testée et le run ne répond pas à la
  question qu'il pose ;
- **si `~/llvq-q4b.llvq` est absent**, le banc échoue — un run sur codes
  synthétiques ne peut pas porter ces seuils, c'est exactement la réserve
  qu'il existe pour lever ;
- **si un bras se révèle éliminé par le compilateur** (sortie non observable,
  boucle constante), sa milliseconde ne mesure rien : le journal doit montrer
  que la sortie de chaque bras a été relue et vérifiée, comme aujourd'hui.

## 7bis. Écarts au protocole — journal, tenu à chaud

*(Chaque entorse s'écrit ici le jour où elle est commise, avec sa raison et
son coût — la règle du 08-10.)*

### É0 — 2026-08-14, correction du lendemain, avant toute ligne de banc

**Ce qui s'est passé.** La version initiale de ce document (commit `09e9654`) a
été soumise à une reconnaissance vérifiée du dépôt — cinq relevés de faits,
chacun repassé par un sceptique indépendant. Elle a trouvé six défauts. Comme
pour l'É0 du [2026-08-11](preregistration-2026-08-11.md), l'ordre aurait dû
être l'inverse : vérifier le dépôt **puis** ancrer. C'est la leçon, et elle se
répète.

**Le défaut principal, et c'est le motif même que ce document dénonçait.** Le
§1.4 interdisait de lire un seuil contre les ancres du **2026-07-31** — et
était **muet sur celles du 2026-08-01**, alors que ce second run (`bin/decreal`,
sol 0,084 · Fixed96 0,152 · Grouped32 0,158 ns/bloc) a tourné sur **blocs
réels du 4B publié**, souffre de la **même** absence de journal dans
`docs/mesures/`, et constitue l'ancrage naturel du banc. J'ai écrit une
interdiction contre un ancrage douteux en en laissant un autre, plus tentant,
grand ouvert à côté.

**Les cinq autres**, tous corrigés dans le texte le même jour :

| § | ce qui était écrit | ce qui est vrai |
|---|---|---|
| 1.2 | le surcoût de soumission traité comme négligeable à 2^24 | le dépôt le chiffre à **12 %** (`format-noyau.md:136-137`) ; il est désormais mesuré par round, sa dispersion imprimée, et il peut suspendre un verdict |
| 1.6 | couverture du tirage non bornée | **286 classes observées**, 0 origine ; 98 des 384 entrées de table jamais exercées par un tirage réel |
| 2 | `masques` = « le chemin servi » | **faux sur Metal** : `grep -rn "Planes" llvq-metal/src/` ne rend rien ; c'est l'ancre historique |
| 2 | bras `cascade-archive` supposé disponible | `unrank_fast` est **privée** (`fastdec.rs:104`) — décision prise ici : on exporte, on ne duplique pas |
| 3.2 | vérification contre `FastDecoder::decode` présentée comme acquise | **strictement nouvelle** côté Metal ; l'existant vérifie contre le transcodeur, chaîne transitive |

**Ce que ça ne change pas.** **Aucun seuil.** Les 1,5 / 2,0 / 0,45 ns sont
intacts, et le régime intermédiaire aussi. Ce qui change est la **discipline de
lecture** (deux jeux d'ancres interdits au lieu d'un, dispersion du surcoût
opposable, couverture déclarée) et deux **décisions de placement** prises
d'avance plutôt que découvertes en codant. Une ligne a été ajoutée à la table
des issues du §6 : le cas où la cascade uniformisée passe le gate CUDA sans
ouvrir P5 — il était déductible des §4.2 et §6 pris ensemble, il est maintenant
écrit.

**Antériorité.** Aucune milliseconde n'existe pour les deux décodeurs neufs, à
cette heure comme la veille : ils n'ont toujours aucune ligne de code. La
correction est donc faite **avant toute mesure**, et la version initiale reste
opposable dans l'historique git (`09e9654`).

### É1 — 2026-08-14, même jour : la clause qui pouvait tout annuler n'avait pas de nombre

**Ce qui s'est passé.** La revue adversariale des pré-enregistrements P2→P5 a
retourné une trouvaille sur **celui-ci** : le §4.3 et le §6 faisaient dépendre
la mort d'E1v de ce que `cascade-archive` « passe la tolérance d'un cadre
capacity-first ». Cette tolérance n'est chiffrée **nulle part** — trois
occurrences de l'expression dans `docs/` et `proofs/`, aucune ne porte de
nombre. Une issue capable d'annuler tout un chantier reposait donc sur un
jugement libre rendu **après** la mesure : celui qui voulait ouvrir P5 aurait
dit qu'elle ne passe pas, celui qui voulait le fermer aurait dit l'inverse.

**Ce qui a été fait.** La clause est chiffrée, **sans introduire de nombre
neuf** : `cascade-archive` ≤ **2,0 ns/bloc**, c'est-à-dire exactement le seuil
que le §4.1 impose déjà à la cascade uniformisée. L'argument est symétrique et
tient en une ligne — un décodeur qui existe, qui est prouvé, qui pèse moins
(2,19 contre 2,3709 b/poids noyau) et qui franchit la barre des décodeurs
neufs ne laisse rien à défendre à un décodeur qu'il faudrait écrire. Le §6
porte désormais **les deux branches**, la haute comme la basse : l'issue
« > 2,0 ns » avait elle aussi été laissée sans action.

**Ce que ça ne change pas.** Aucun autre seuil, et pas davantage la valeur de
2,0 ns, qui était déjà dans le document. Ce qui change est qu'une issue
qualitative devient une issue chiffrée, décidée d'avance, par un nombre que le
document portait déjà.

**Pourquoi ne pas avoir chiffré la tolérance elle-même.** Parce qu'inventer
ici un seuil « capacity-first » aurait été poser un critère produit dans un
document technique, alors que le triplet dont il dépend (carte, contexte,
marge, format KV) n'est pas arbitré — la note produit du 2026-08-13 a ses
cases vides, et sa table §B ne se reproduit d'ailleurs pas. Un seuil hérité
d'un document non arbitré aurait déplacé la porte de sortie, pas fermée.

### É2 — 2026-08-14, avant la première ligne du banc : un V0 que le bras ne pouvait pas franchir

**Ce qui s'est passé.** En préparant l'écriture des références CPU, la question
« contre quoi chaque bras se vérifie » a été reprise à la lettre. Le §3.2
exigeait des **deux** décodeurs neufs une égalité bloc à bloc avec
`FastDecoder::decode`. C'est juste pour la cascade uniformisée, qui décode
l'ordre d'archive. C'est **impossible** pour la marche binomiale, qui décode
un ordre combinatoire : les relier, c'est la ré-bijection CNS, c'est-à-dire
P5 — que ce banc conditionne. Le V0 du bras exigeait donc l'existence de ce
que son propre verdict autorise.

**Ce qui a été fait.** Deux étalons distincts, tabulés au §3 : égalité point à
point avec `fd.decode` pour les deux cascades ; aller-retour sur sa propre
bijection **plus** preuve de bijection (distinction deux à deux, compte égal à
la cardinalité de la classe) pour la marche binomiale. Et la réserve qui va
avec : n'ayant pas de transcodeur, ce bras est alimenté par des rangs tirés
uniformément dans la cardinalité de la classe **réelle** de chaque bloc réel —
donc classe et magnitude de rang sont exercées à leur distribution vraie, mais
le bras **ne prouve pas** que le format E1v décode le modèle publié. Cette
preuve est `C2` de P5.

**Ce que ça ne change pas.** Aucun seuil. Aucune issue. Ce qui change est
qu'un des deux bras jugés avait un V0 infranchissable, donc un seuil
inatteignable par un chemin qui n'était pas le sien.

**Ce que ça apprend.** Le défaut n'était pas dans un chiffre mais dans une
**phrase qui paraissait plus rigoureuse que la bonne** : exiger le même étalon
des deux bras semblait plus strict, et c'était en réalité une confusion entre
deux objets — un rang de permutation et un rang combinatoire. La rigueur
uniforme n'est pas la rigueur.

### É3 — 2026-08-15, le banc est écrit et n'a pas tourné : les trois arbitrages que le plan refusait d'inventer

> 🚨 **PROPOSITION, non acquise. Elle attend l'arbitrage de l'opérateur, et le
> tampon doit venir APRÈS lui.** Une fois `ots stamp` posé, ce document est en
> lecture seule pour toujours : un É3 stampé est un engagement, un É3 écrit
> après la première milliseconde n'est plus un pré-enregistrement.

**Ce qui s'est passé.** Le plan d'implémentation du banc a relevé **trois
décisions qui ne sont dans aucun pré-enregistrement** et a refusé de les
prendre : *« aucun ne s'invente ici »*. Il avait raison, et il avait aussi
raison de dire quand elles doivent se prendre — **avant la première mesure,
parce qu'après elles deviennent négociables**. Le banc est écrit et n'a pas
tourné ; c'est le dernier moment où elles sont encore des règles plutôt que
des interprétations.

#### (a) Un sixième bras, `sol-rang` — **proposé : oui**

Le §2 fixe **cinq** bras. En ajouter un est un écart, et c'est pourquoi il
s'écrit ici.

**Le défaut qu'il corrige.** Les bras ne lisent pas le même nombre d'octets :
`sol` et `masques` lisent les 12 octets de `Fixed96`, les deux cascades en
lisent 8, la marche 12. **`sol` est donc le plancher de `masques` et de
personne d'autre.** Sans sixième bras, le journal doit porter une réserve en
prose — *« un décodeur de rang qui bat le sol sur le temps ne bat pas le sol
sur le travail »* — et une réserve en prose est ce qui disparaît d'une citation
au troisième réemploi. C'est le motif que ce dossier documente partout ailleurs.

**Ce qu'il est.** Il lit les 8 octets du flux de rang — **le même buffer que
les bras 2 et 3** — et ne décode rien. Le plancher des bras de rang, mesuré au
lieu d'être supposé.

**Où il se place.** **En dernière position**, index 5, pour ne réordonner aucun
bras existant (§1.3). Le rapport `× le sol` reste formé contre le bras 0, qui
ne bouge pas ; le bras 5 s'y lit comme les autres.

**Ce que ça coûte.** Un dispatch de plus par round, sur un noyau de vitesse
plancher : quelques millisecondes sur les 18 rounds. Zéro dollar. **Ce que ça
ne coûte pas : aucun seuil ne bouge.** Les 1,5 / 2,0 / 0,45 ns restent lus
contre les mêmes bras.

#### (b) La règle de suspension appliquée aux seuils absolus — **proposé : oui, à l'identique**

Le §1.2 amendé dit : *si l'étendue du surcoût dépasse la moitié de l'écart
entre deux bras que le verdict sépare, ce verdict n'est pas rendu.* Il parle
d'un écart **entre deux bras**. Or les trois seuils du §4 sont des
comparaisons **à une constante**, et le §1.2 est muet sur ce cas.

**Proposition : même forme, même constante.** Un verdict de seuil n'est pas
rendu si `étendue_ns > |ns_bras − seuil| / 2`.

**Pourquoi maintenant et pas après.** Sans cette ligne, un bras à 0,44 ns
devant un gate à 0,45 avec un surcoût dont l'étendue vaut 0,10 ns se discute
après coup, et il se discutera dans le sens que voudra celui qui parle. C'est
exactement la porte de sortie que l'É1 a fermée sur la clause E1v — le même
défaut, à un autre endroit du même document.

**Le sens de l'erreur est assumé** : la règle ne peut que **suspendre** un
verdict, jamais en fabriquer un. Elle est donc conservatrice par construction,
et elle coûte un run de plus quand elle mord.

#### (c) Un critère chiffré d'acceptation du tirage — **proposé**

Le §7 dit qu'un tirage dont l'histogramme *s'écarte* de celui du fichier ne
répond pas à la question posée. Il ne chiffre pas « s'écarte ».

**Proposition, en trois clauses :**

1. **Le `z` par classe** est `(observé − attendu)/√attendu`, avec
   `attendu = f_classe · N / total`.
2. **Le maximum est pris sur les classes dont l'attendu est ≥ 25** — en
   dessous, l'approximation normale ne tient pas et le `z` n'a pas de sens. Le
   nombre de classes écartées à ce titre est **imprimé**, jamais absorbé.
3. **Le tirage est refusé si `max |z| > 4,0`**, ou si une classe d'attendu ≥ 5
   ressort **vide** du tirage.

**D'où sort le 4,0**, et ses deux réserves, parce qu'un seuil sans dérivation
est un seuil qu'on déplacera : sur ~286 classes, `P(|Z| > 4)` vaut 6,3·10⁻⁵ par
classe, soit **moins de 2 % de fausse alarme** sur l'ensemble — un budget
acceptable pour un garde qui doit rester vert sur un bon tirage. Réserve n°1 :
le tirage étant **sans remise**, la loi exacte est hypergéométrique, dont
l'écart-type vaut `√(attendu·(1−N/total))` — le `z` imprimé, qui divise par
`√attendu`, **sous-estime donc l'écart vrai d'un facteur ≈ 0,94**. Le test est
~6 % moins sensible que son nominal, et sa fausse alarme réelle est sous 1 %.
Réserve n°2 : ce critère juge la **composition en classes** du tirage, et rien
d'autre — il ne dit rien de la corrélation entre blocs voisins, qu'un
échantillonnage par réservoir ne peut de toute façon pas introduire.

**Ce que ça ne change pas.** Aucun seuil du §4, aucune issue du §6, aucun bras.
Ce sont trois règles de **lecture**, et les trois vont dans le sens qui coûte
au banc : un bras de plus à battre, un verdict qui peut être suspendu, un
tirage qui peut être refusé.

**Antériorité.** Aucune milliseconde n'existe pour aucun des cinq bras au
moment où ceci est écrit : `bin/rankbench` refuse de démarrer tant que
`proofs/preregistration-p1-2026-08-13.md.ots` est absent, et il l'est.

## 8. Ce qui est connu à la signature — divulgation datée

- **Aucune milliseconde n'existe** pour la cascade uniformisée ni pour la
  marche binomiale, sur aucun matériel. Aucun compte de registres, aucun
  profil. Le profileur n'a jamais servi sur ce projet.
- **`e1v-séparé` est vert en largeur** (53,332 bits/bloc → 2,3709 b/poids sous
  warp-scan) et **égal à `radix2` au bit près** (test
  `e1v_split_is_radix2_bit_for_bit`). Sa largeur était déjà publiée le 08-12
  sous le critère de 2,60 et classée ❌ sur le seul booléen shift-only : ce
  banc juge ce qui manquait, le décodeur, pas la place.
- **L'archive fait 2,19 b/poids** et son décodeur existe et est prouvé. C'est
  le concurrent que tout bras neuf doit battre, et il est dans le banc.
- Le seuil de profondeur ≤ 24 du spec X4 **n'est pas rouvert par ce
  document** : c'est une décision de passation, prise en P5, et P5 ne s'ouvre
  que si la **marche binomiale** passe 0,45 ns — pas « si le banc est vert »,
  cf. la table du §6.
- **Divulgué au titre de l'É0 (2026-08-14)** : les trois chiffres du run
  `decreal` du 2026-08-01 sont connus de l'auteur — sol 0,084 · Fixed96 0,152 ·
  Grouped32 0,158 ns/bloc, sur blocs réels. Ils sont **écrits ici** précisément
  pour qu'on ne puisse pas dire après coup qu'ils ont inspiré un seuil : les
  seuils datent du 2026-08-13 et ne bougent pas, et le §1.4 interdit de les
  lire contre ces trois nombres.
