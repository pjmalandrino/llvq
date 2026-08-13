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

**1.2 — `N = 2^24` est un minimum, pas un choix libre.** À 2 M blocs le
surcoût de soumission valait le travail mesuré ; c'est l'un des trois défauts
qui faisaient dire « 25 tok/s, c'est mort » avant correction. Tout bras
mesuré sur moins de `2^24` blocs est nul et non avenu.

**1.3 — Tous les bras dans un seul processus, un ordre de dispatch fixe,
tous dispatchés à chaque round.** Rapports formés **round par round**, jamais
comme quotient de deux minima issus de rounds n'ayant jamais coexisté (règle
de maison n°2). Un bras ajouté ne réordonne jamais les bras existants.

**1.4 — Les ancres se remesurent dans le même run, et les seuils se lisent
contre elles.** `sol` et `masques` sont redispatchés à chaque round. **Aucun
seuil, aucun rapport, aucune conclusion ne se lit contre les 0,08 / 0,11 /
8,27 ns du 2026-07-31** — un run sans journal, sur une autre distribution,
sur une machine dont l'état de trois mois plus tôt n'est pas reconstituable.
Ces trois nombres ne sont dans ce document que comme contexte historique.

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

## 2. Les bras, figés ici

Cinq, tous dans le même processus :

| bras | rôle | existe ? |
|---|---|---|
| `sol` | aucun décodage — lit les octets, accumule un produit scalaire | ✅ ancre |
| `masques` | masques imbriqués, le chemin servi | ✅ ancre |
| `cascade-archive` | l'unranking du format d'archive tel quel (récurrence u64 `M' = M·c/n`) | ✅ **étalon, pas décoration** — voir §4.3 |
| `cascade-uniformisée` | 24 itérations identiques pour toute classe, réciproques magiques par (classe, étage) en table, candidats en ILP, sélection branchless, **zéro indexation dynamique** | ❌ à écrire |
| `marche-binomiale` | le décodeur d'E1v : unranking par table `C(n≤24, k≤12)` tenue en L1, pas à compte fixe, **zéro division** | ❌ à écrire |

Chaque bras **accumule un produit scalaire et écrit un float par bloc** — la
forme d'un noyau fusé. Aucun bras n'écrit les 24 poids décodés : c'est le
défaut n°1 du banc de 2026-07-31, qui mesurait ses propres écritures non
coalescées (« un banc mémoire déguisé en décodeur »). L'activation est chargée
une fois par threadgroup, jamais relue par itération (défaut n°2).

## 3. V0 avant V1 — l'exactitude d'abord, sans exception

**Aucune milliseconde n'est chronométrée avant que le décodeur soit prouvé.**
Dans cet ordre :

1. **Référence CPU écrite indépendamment du shader**, comme
   `decode_masks_cpu` l'est aujourd'hui — pour qu'un malentendu partagé ne
   passe pas.
2. **Bloc à bloc contre `FastDecoder::decode`** (`llvq-search/src/fastdec.rs:243`)
   sur un échantillon large et couvrant toutes les classes du tirage.
3. **Sweep intégral des 150 681 600 blocs** du 4B scellé, harnais du sweep
   E1c (`llvq-artifact/tests/`), compte de blocs imprimé — pas un skip.

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
- **L'étalon `cascade-archive` peut fermer la ligne par le haut, et c'est
  posé d'avance** : si la cascade d'archive telle quelle passe la tolérance
  d'un cadre capacity-first, **E1v est mort-né** — l'archive existe déjà, elle
  fait 2,19 b/poids contre 2,37 pour E1v, et un décodeur qu'on n'a pas à
  écrire bat un décodeur qu'on doit prouver sur 150,7 M blocs. Ce bras n'est
  pas un contrôle décoratif : c'est le seul du lot qui puisse rendre tout le
  reste inutile.

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
| marche binomiale ∈ ]0,45 ; 1,5] ns | E1v survit en largeur et **reste sans chemin d'exécution** ; ni P5 ni bras CUDA — la publication le dit comme point de courbe |
| marche binomiale > 1,5 ns | **E1v est mort comme format servi** ; sa largeur de 2,3709 reste un résultat de comptage, rien de plus |
| cascade uniformisée ≤ 2,0 ns | la famille cascade reste candidate pour le chemin d'archive |
| cascade uniformisée > 2,0 ns | la famille cascade se referme sur Metal |
| `cascade-archive` acceptable en cadre capacity-first | **E1v mort-né** (§4.3) — l'archive fait mieux en b/poids et existe |
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

*Aucun à ce jour.*

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
  que si ce banc est vert.
