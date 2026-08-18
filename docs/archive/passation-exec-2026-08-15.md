# Passation — session d'exécution du 2026-08-14/15

> **Pour la session qui reprend.** Ce document est autonome. Il remplace
> [`passation-exec-2026-08-13.md`](passation-exec-2026-08-13.md), dont les
> seuils P2/P4 sont **faux** (voir §5) et dont le chantier MoE est en pause.
>
> 18 commits, `3879cde..ea2d9aa`. **0 $ dépensé** — aucun job GPU loué.
> ~2 h 45 de Mac de mesure, le reste en code et en documents.

## 1. Ce qui a changé, en dix lignes

- **Cinq pré-enregistrements existent** (`proofs/preregistration-p{1..5}-*.md`).
  P1 a traversé **trois amendements** (É0, É1, É2) avant sa première mesure ;
  P2→P5 ont été écrits, passés à une revue adversariale qui a trouvé
  **18 bloquants**, puis réécrits.
- **P3 est mesuré et clos** : le cache KV à 8,5 bits ne coûte pas de qualité et
  **n'est pas servi par défaut**. Verdict « contexte court seulement ».
- **P1 a ses deux décodeurs, ses deux shaders, et V0 est VERT** sur 1 048 576
  blocs réels par bras. Il reste le banc et les seuils.
- **Le chantier MoE (P2, P6) est en PAUSE**, décision de l'opérateur. Modèle
  tranché : **Qwen3-30B-A3B**, gpt-oss écarté.
- **P4, P5, P7 n'ont pas bougé** — ni code, ni mesure.

## 2. Où reprendre — P1, et rien d'autre par défaut

C'est le chemin critique, et il est à deux pas de son premier chiffre.

**Ce qui existe** :

| | |
|---|---|
| `llvq-search/src/rankdec.rs` | les deux références CPU, 5 tests, 0 warning |
| `llvq-metal/shaders/{cascade_uniform,binomial_walk}.metal` | les deux shaders, **compilent** |
| `llvq-metal/src/p1host.rs` | les trois miroirs `#[repr(C)]` et leurs constructeurs |
| `llvq-metal/src/bin/mslcheck.rs` | verdict de compilation MSL en 3 s |
| `llvq-metal/src/bin/p1v0.rs` | **V0, vert sur les deux bras** |

**Ce qui manque, dans l'ordre du §3 du pré-enregistrement** :

1. **La fixture synthétique** — l'origine et la coquille 13. Le tirage réel
   touche **243 classes sur 383** : le 4B est cap 12 et n'a **aucun bloc
   origine**, donc 141 entrées de table ne sont jamais exercées. Forme à
   copier : `llvq-artifact/tests/e1c_format.rs:81-96` (`fixture_indices`).
2. **Le sweep intégral** des 150 681 600 blocs, dans `llvq-artifact/tests/`
   — seul endroit d'où l'on peut ouvrir un `.llvq`.
3. **L'aller-retour rang → arrangement → rang côté GPU** pour la marche. Il
   existe côté CPU (`the_walk_round_trips_on_its_own_bijection`) et ne borne
   rien sur le shader.
4. **Le banc à 5 bras**, puis V1.

⚠️ **`Kernel::time` ne peut pas satisfaire le §1.3** : il boucle warmup+reps
sur **un seul bras**, ce qui produit exactement le quotient de deux minima que
la règle de maison n°2 interdit. Le seul gabarit conforme du dépôt est la
boucle manuelle de `thesis.rs:871-901`. Un plan de banc détaillé (667 lignes)
est dans le scratchpad de la session ; s'il a disparu, il se réécrit depuis le
§1 du pré-enregistrement.

⚠️ **L'overhead vaut 12 % à 2^24** — le dépôt le chiffre lui-même
(`format-noyau.md:136-137`). Il se mesure **à chaque round**, sa dispersion
s'imprime, et si son étendue dépasse la moitié de l'écart que le verdict
sépare, **le verdict n'est pas rendu** (É0).

## 3. Les seuils de P1, et ce qu'ils décident

| bras | vert | rouge |
|---|---|---|
| `marche-binomiale` | ≤ **1,5 ns/bloc** | > 1,5 ⇒ mort |
| `cascade-uniformisée` | ≤ **2,0 ns/bloc** | > 2,0 ⇒ mort |
| gate CUDA de P4 | meilleur des deux ≤ **0,45 ns** | sinon aucun job |
| `cascade-archive` ≤ **2,0 ns** | ⇒ **E1v mort-né**, P5 ne s'ouvre pas | — |

**P5 s'ouvre si et seulement si la MARCHE passe 0,45 ns** — pas « si P1 est
vert ». Une cascade verte à 0,4 ns avec une marche à 1,0 autoriserait le bras
CUDA **sans** ouvrir P5. C'est le seul cas où les deux règles divergent, et il
est écrit dans la table des issues.

## 4. P3 — clos, et ce qu'il laisse ouvert

**Mesuré** ([`mesures/kvq8-4b-2026-08-15.txt`](../mesures/kvq8-4b-2026-08-15.txt)) :
perplexité **+0,049 %** [−0,071 ; +0,170], MMLU **+0,33 pp** [−0,45 ; +1,22],
les deux intervalles contenant zéro. Débit **0,927× et 0,945×** à `n_new` 128.

**Non servi par défaut** : la série `n_new = 1024` a été abandonnée en entier
(première invocation **661 s > 600 s**), le §4.3 exige le vert sur les quatre
séries, donc le verdict est étiqueté « contexte court seulement ».

**Ce qui reste la question produit** : le contexte long. À `n_new = 1024` le
bras f16 tombe à 5,6 tok/s contre 9,6 à 128 — le coût du cache y domine, donc
c'est là que l'allègement devrait payer, et c'est la seule région
inaccessible. ⚠️ **Ne pas rouvrir en relisant le même run** : les deux séries à
128 sont rendues et vertes, le manque est une région non visitée, pas une
imprécision. Il faut un instrument qui garde le modèle résident entre les deux
bras — `gbench` charge un modèle par processus.

**Le gain mémoire n'a pas été mesuré non plus** : `ppl` et `mmlu` créent un
`KvCache` frais par bloc et tiennent 16,8 Mo, pas 0,604 Go. Le gain est un
**compte** : 147 456 → 78 336 o/token, **÷1,882 et pas ÷2**, géométrie
36 couches × 8 têtes KV × head_dim 128, batch 1. Il se cite en **octets/token
avec sa géométrie**, jamais en b/param — un cache n'est pas un paramètre.

## 5. 🚨 Trois seuils hérités étaient FAUX — ne pas les reprendre

La passation du 13 les porte encore. Ils sont corrigés dans les
pré-enregistrements, pas dans elle.

| hérité | pourquoi il est faux | valeur juste |
|---|---|---|
| « kill ppl > **0,7 %** » | bruit de **graine de calibration** entre fichiers *différents*, sans objet pour un A/B à fichier constant | intervalle t **apparié** fenêtre par fenêtre : **±0,12 %** mesuré, quatorze fois plus serré |
| « σ McNemar **0,4-0,6 pp** » | **jamais calculé**, et porte sur le taux poolé non pondéré | SE appariée **mesurée : 0,43 pp** ici, 0,79-1,44 pp entre modèles différents |
| **K2** de P4 : « `T(k=8) ≤ 0,60 × T(k=1)` » | **arithmétiquement impassable** — un noyau à k=8 fait 8× les FMA et 8× les stores | **par colonne** : `T(k=8)/8 ≤ 0,60 × T(k=1)`, soit `T(k=8) ≤ 4,80 × T(k=1)` |

Et **K1 était écrit à l'envers** dans le brouillon de P4 : il se lit sur le
**rapport vs FP16**, pas sur le temps. Sur les médianes du 08-11 les deux
formes concluent l'inverse l'une de l'autre.

## 6. Le budget de P4, corrigé d'un facteur 2

« ~0,3-0,5 $ » est faux. Historique mesuré (`docs/data/jobs.csv`) : tout job
`planesbench` à 5 bras ou plus coûte **0,77-0,78 $**, parce que le transcodage
hôte pèse **1 468-1 481 s avant le premier round** — et il ne se réduit pas en
désélectionnant, le transcode Slot32 étant inconditionnel. Budget réel :
**0,8-1,0 $**.

⚠️ Et **trois bras du plan n'ont aucun code** : cuBLAS (le dénominateur
publiable ; `tv_f16` est maison et ne peut pas l'être), le noyau **E1c CUDA**
(le « banc gelé à 0,2 $ » du CLAUDE.md n'existe pas), et le **support k
colonnes**. Plus le **chronométrage par forme** via events CUDA, sans lequel
K2 n'est attribuable à rien.

## 7. MoE — en pause, modèle tranché

Décision de l'opérateur du 2026-08-14 : **P2 et P6 en pause**, plan conservé.

**gpt-oss-20b est écarté** — il rate C1 (MXFP4 natif, aucune référence f16) et
sa forme est la plus favorable du menu (32 experts top-4, **12,5 % actifs**)
quand la cible tourne à 3-4 %. Il n'était retenu que pour la continuité du
contrôle avec le dump agrégé du 08-12 : un instrument, pas un objet.

**Retenu : Qwen3-30B-A3B** (128 experts, top-8, 6,3 % actifs, f16 propre, même
famille que tout le dépôt), pour **P2 et P6 à la fois**.

Coûts projetés sur l'ancrage **mesuré** du 8B (275 min / 12,61 $ / 6,95 Md sur
`rtx-pro-6000`), pas sur les cœur-heures de l'étude :

| | coût | note |
|---|---|---|
| P2 routage, 30B-A3B | ~1,4 $ | **décide de P6 — rapport 1:50** |
| P2-b, Qwen3-Next-80B-A3B | ~5,5 $ | 512 experts, 3,8 % actifs — rend le vert **transportable** |
| P6 quantification, 30B-A3B | **~69 $** | 579 cœur-h ; l'étude disait 25-55 $, c'est le prix du cœur-heure qui diffère |

**Deux faits de tarification à réutiliser** : `rtx-pro-6000x2` coûte le **même
prix par cœur-heure** que le x1 (0,1196) — la moitié du temps mur est
**gratuite**, et personne ne s'en était servi. `l4x4` est le cœur-heure le
moins cher avec GPU (0,0792, −34 %), sous réserve que 24 Go de VRAM suffisent.

`ops/moe_lru.py` est écrit (stdlib seule, cinq cas de V0.4 verts, mutation LRU→FIFO
tuée). Le drapeau `--trace` de `ops/moe_routing.py` reste **à écrire** :
l'information temporelle est détruite **dans le hook**, ce n'est pas un
paramètre à ajouter.

À la reprise, l'amendement É0 de P2 doit changer le §2.2 (ligne de commande
figée sur gpt-oss), le §4.3 (ordre des critères) et le V0.1 (le contrôle
contre le dump du 12 disparaît avec gpt-oss).

## 8. Ce qui vous revient — deux gestes, et le premier bloque P4

1. **`ots stamp` sur les cinq pré-enregistrements.** `ots` n'est pas installé
   sur la machine. P1 a bougé trois fois depuis son ancrage ; il est stable
   maintenant. À poser **avant la première milliseconde**, sinon P1 hérite de
   la dette de provenance qu'il reproche au lot du 13.
2. **L'arbitrage de `docs/note-produit-2026-08-13.md`** — 6 cases, toujours
   non commité. Sans le triplet, **P4 n'a pas de critère d'admission complet**
   et `S_alt` n'a aucun statut. ⚠️ Sa table §B **ne se reproduit pas** :
   « 16k / KV f16 / marge 5 » calcule 2,26 et imprime 2,27, et aucune
   convention unique ne rend ses neuf cellules. L'unité de « 32 Go » (GiB ou
   décimal) n'y est définie nulle part — ±0,28 b/poids, plus que ce que le
   passage KV f16→q8 achète.

## 9. Dettes ouvertes, et une qui n'est pas de cette session

- **`g6_pack` échoue en debug, passe en release** — vérifié au commit
  `3879cde`, donc **antérieur** à cette session. Motif classique d'un décalage
  de 64 bits que la release absorbe en silence. Une tâche est ouverte.
- **`xcrun metal` n'est pas installé** : les shaders ne sont validés que par le
  compilateur **runtime** (`bin/mslcheck`). C'est celui que le banc utilise,
  donc c'est suffisant — mais aucun `.metallib` n'est produit hors ligne.
- **Les chemins de division 1 et 2** du shader de cascade (inverse modulaire,
  découpe q/r) ont leurs tables construites et testées côté Rust, mais
  **n'ont jamais traversé le compilateur Metal**. Seul `LLVQ_CASCADE_DIV == 0`
  est le bras pré-enregistré ; changer ce défaut relève d'une entrée §7bis.
- **La composition des classes paires** pour la marche : le bras décode **une**
  marche de 24 créneaux ; comment une classe paire d'archive se compose (deux
  unrankings contre un mot de Golay + réparation de parité) est **ABSENT**. Un
  journal de vitesse devra dire si le coût rendu est celui d'une marche ou d'un
  bloc.
- **L'écart Metal ↔ CUDA du lot A** est déplacé, pas refermé : le bras f16 de
  P3 reproduit **56,09 %**, la valeur Metal du 08-02, à trois mois d'écart. Le
  harnais Metal ne dérive donc pas ; l'écart avec le 55,59 CUDA est de
  backend. Le refermer demande un rejeu CUDA.

## 10. Ce que cette session apprend, et qui vaut au-delà d'elle

**Cinq « verts vides » ont été trouvés, tous du même genre : un test qui passe
sur du code faux.**

| où | ce que le vert cachait |
|---|---|
| P3, avant le contrôle positif | un bras q8 non branché rend Δppl = 0, ΔMMLU = 0, débit 1,00× **et** `LLVQ_VERIFY_CACHE` vert — trois verts vides |
| `binomial_walk` | il ne décodait que **50,02 %** du codebook, et les trois tests filtraient exactement les entrées où il échoue |
| dernier genre de la marche | une mutation écrivant `j+1` **survivait aux cinq tests** — l'aller-retour ne peut pas voir un genre qui n'a pas de rang |
| `verify` de p1v0 | un noyau rendant **NaN** imprimait VERT, avec un pire écart **plus flatteur** qu'un noyau juste |
| `ends` de p1host | la mutation `ends + 1` survivait aux 5 tests **et** à V0 : deux accesseurs, rien qui les recolle |

Et le plus instructif est ailleurs : le test `the_step_count_depends_only_on_the_class`
**ne pouvait pas échouer** sur la propriété qu'il nomme, parce que son jumeau
instrumenté comptait ce que la marche *aurait dû* faire. C'est exactement le
motif du §5 du dossier — *une assertion qui n'exerce pas le paramètre qu'elle
est censée couvrir* — et il a été écrit en croyant faire l'inverse.

**La règle qui en sort** : quand un test garde une propriété, muter le code
**et** vérifier que le test tombe. Un test écrit le même jour que le code qu'il
garde partage ses angles morts.

⚠️ **Et muter se fait sur un fichier COMMITÉ.** `git checkout` restaure à HEAD :
sur un fichier suivi mais non commité, il emporte le travail avec le mutant
(payé une fois cette session) ; sur un fichier non suivi, il ne restaure rien
et les mutants s'empilent. Pour un fichier neuf, sauvegarder à part et
restaurer par copie.
