# Pré-enregistrement — P4 : le job carte mutualisé (balayage en k, X3/E1c), et les seuils qui l'autorisent ou le ferment

**Date : 2026-08-14.** Écrit **avant toute mesure**, et — différence avec les
trois précédents — **avant la première ligne du code qu'il juge**. À cette
heure :

- **aucun appel cuBLAS** :
  `grep -rniIE 'cublas|CudaBlas' --include='*.rs' --include='*.cu' --include='*.cuh' --include='*.toml' --exclude-dir=.claude .`
  rend quatre lignes — un commentaire de `llvq-cuda/Cargo.toml:23`, deux entrées
  de la liste de features `cudarc` (`:54-55`, recopiée de candle, jamais
  utilisée), un commentaire de `llvq-llm/src/bin/smoke.rs:58`. ⚠️
  `--exclude-dir=.claude` est **nécessaire** : une copie de travail sous
  `.claude/worktrees/` double ces quatre lignes. Le dénominateur f16 du banc est
  `tv_f16`, écrit à la main (`llvq-cuda/kernels/matvec.cu:407-441`) ;
- **aucun noyau ni bras E1c côté CUDA** : `grep -rniI e1c llvq-cuda/` rend zéro
  ligne, `ls llvq-cuda/kernels/` rend 15 fichiers, aucun `e1c*.cu`. L'existant
  est **purement hôte** (`llvq-artifact/src/e1c.rs`, ses tests, `rtbits.rs`) ;
- **aucun support de k dans aucun noyau maison** : `tv_planes`
  (`llvq-cuda/kernels/planes.cu:27-35`), `tv_planes12x`, `tv_golay70`, `tv_f16`
  n'ont ni k ni stride de colonne ; seul `awq_gemv.cu:116-119` porte un
  `blockIdx.z`, lancé à z = 1 (`planesbench.rs:700-708`). Le chemin servi déclare
  ne pas savoir faire plus d'un token par appel (`fused_cuda.rs:18-23`) ;
- **aucun pré-enregistrement P4** (`ls proofs/` : README, 08-10, 08-11, 08-13,
  p1-08-13). Les seuls seuils P4 vivent dans une passation
  (`docs/archive/passation-exec-2026-08-13.md:162-168`), donc pas opposables.

🚨 **Le CLAUDE.md laisse croire que « le bras de banc CUDA est un pur test de
vitesse à ~0,2 $ ». Le code n'existe pas** : « X3 — bras de banc CUDA et noyau
`e1c.cu` | ❌ pas écrit » (`passation-lot-x-2026-08-12.md:205`, idem
`runbook-lot-x-mac.md:19-20`, `spec-memoire-extreme-2026-08-12.md:7`).

> Hérite **sans dérogation** des gardes du préreg du
> [2026-08-10](preregistration-2026-08-10.md) (§7), de sa comptabilité (§6) et
> de sa règle de résolution (§4). ⚠️ **Exception d'héritage, complète** : le §6
> du 08-10 (`:219`) annonce **deux** corrections — la colonne `bpw_payload` du
> CSV et le « Payload rates » de `paper/sections/layouts.tex:36`. **Les deux
> sont faites** (la colonne s'appelle `bpw_kernel`,
> `docs/data/echelle-formats.csv:1` ; `grep -rniI 'Payload rates' paper/` est
> vide au 2026-08-14). ⚠️ Ni signé ni horodaté tant que l'opérateur ne l'a pas
> fait. **Le tampon est porteur** : demandé avant le premier noyau k, et
> **avant le go de dépense**.

---

## 0. Ce que ce job peut et ne peut pas conclure

**Ce qu'il mesure** : des temps de matvec sur **une** L40S, sur les 252
matrices du **4B publié**, dans **un** processus, activations **synthétiques**
(16 384 f32 tirés une fois, `planesbench.rs:888-890`). Pas un token ; aucun
tok/s ne s'en déduit.

1. **Aucun bras ne peut être dit « saturer la mémoire ».** Pas de sol de bande
   passante dans `planesbench` — `tv_floor_*` vivent dans `matvec.rs:724-728`, et
   `preflight` ne charge `floor_probe` que pour ses registres (`:115`). La plaque
   de 864 Go/s est **interdite** pour un « % du pic » (ECC, `preflight.rs:103-107`) ;
   tout « % de la borne » se forme contre le FP16 du même run
   (`docs/data/README.md:7-15`).
2. **Aucune attribution par profil** : `ncu` **absent du conteneur**
   (`cuda-preflight-2026-08-05.txt:13`). Reste le comptage d'octets — déjà
   **optimiste** sur ce noyau (Golay70 v2 : 1,9–2,4× estimé, **1,77× mesuré**).
3. **L'occupation n'est pas lisible** : `grep -niI occupanc llvq-cuda/src/gpu.rs`
   est vide ; les « ~852 blocs résidents » (142 SM × 6) sont un **compte de
   journal** (`attribution-cuda-2026-08-05.txt:96`). `num_regs` **ne la borne
   pas**.
4. **Rien ici ne porte sur la cible MoE.** L'addendum du spec
   (`spec-memoire-extreme-2026-08-12.md:17-27`) et l'étude qui le fonde
   (`etude-moe-memoire-extreme-2026-08-12.md:179-181`) placent le critère 1,6×
   hors périmètre MoE et y admettent `Golay70` « de fait », sur une tolérance
   dite *capacity-first*. 🚨 **Cette tolérance n'est chiffrée nulle part** —
   trois occurrences, aucune avec un nombre (§8). **Aucune fermeture de ce
   document ne vaut pour le MoE, ni dans un sens ni dans l'autre.**
5. **Un vert n'achète que la suite du dossier.** Un rouge ferme : c'est là
   qu'est la rentabilité.

## 1. La comptabilité, figée ici

**1.1 — Octets en comptabilité NOYAU** (§6 du 08-10) : flux + queue f32 +
échelles de ligne f32, sur **tous** les poids de la matrice, queue comprise
(`rtbits.rs:316-326`, doc `:72-80`). Ni payload, ni b/param.

**1.2 — Toute comparaison MÉMOIRE en b/param MODÈLE ENTIER**, embedding
compris, q8 facturé 8,5 b/param (préreg 08-11 §2.1). Jamais un b/poids de
projections contre un b/param de modèle entier (errata du lot A).

**1.3 — Quelle colonne juge.** Le banc imprime trois conventions dans le même
log ; ne pas trancher, c'est laisser choisir après coup.

| grandeur | convention | juge ? |
|---|---|---|
| « X× vs FP16 », rapports entre bras | **médiane des rapports round par round**, avec étendue (`planesbench.rs:1699-1701`, `:1771`, `:1790-1799`) | ✅ **elle seule porte K1, K2, X3** |
| « Go/s » | `octets / temps MINIMUM` (`:1753`, `:1763`) | ❌ descriptive |
| « avec le lm_head f16 » | **minima des deux côtés** (`:1851`, `:1857`) | ❌ et ce n'est pas une mesure du lm_head : 389 070 848 × 2 o divisés par le débit FP16 (`:1849`) |

**1.4 — À k > 1, « Go/s » et « b/poids » ne sont PAS publiés pour les bras k.**
`arm_bytes` (`:399-409`) ne facture que les poids ; à k colonnes ils ne bougent
pas et le trafic d'activation est multiplié par k. Le journal imprime alors le
**temps** (min, méd, max), les **rapports**, et le trafic d'activation
**calculé** à côté (précédent : `six-arm-awq-2026-08-10.txt:26-32`). Le b/poids
noyau reste imprimé **à sa valeur de k = 1**, étiqueté propriété de format.

**1.5 — Le garde L2 ne protège rien à k > 1.** `reps` n'existe que sur le
chemin **synthétique** (`:833-837`, usage unique `:1384-1385`) ; sur le modèle
réel la passe *est* les 252 matrices, sans répétition, et la garde se réduit à
une ligne (`:1828-1837`). L'activation pèse ≤ 39 Ko à k = 1 contre 100 663 296 o
de L2 lus au driver (`cuda-preflight-2026-08-05.txt:28-29`) : servie par le
cache à tout k mesuré. **Un bras k mesure l'amortissement d'un poids relu, pas
une propriété DRAM.**

**1.6 — Le fichier est nommé, son absence ÉCHOUE.** Le banc lit
`/model/qwen3-4b-llvq.bin` (252 matrices, 3,63 Md de poids, 150 681 600 blocs).
🚨 **Sans argument de chemin il ne s'arrête pas : il retombe sur le chemin
synthétique** (`args().nth(1)`, branche `None`, `:1384-1385`) et imprime une
table d'allure normale. Le journal doit porter « *252 matrices, 3.63 Md de
poids, en N s* » et le nom du fichier ; **sinon le run est synthétique, nul et
non avenu** — pas « à interpréter avec prudence », nul.

## 2. Le protocole, figé ici

**2.1 — Invariants hérités** : une L40S, un processus, bras entrelacés dans
chaque round, ordre de dispatch fixe, **7 rounds dont 2 jetés** (`:111-112`),
rapports **round par round**, jamais un quotient de deux minima.

**2.2 — Règle de décision.** `R = max(Δ_contrôle, demi-étendue intra-run du
bras le plus dispersé)` ; `|Δ| > 2R` = **séparation** ; `|Δ| < R` =
**indiscernable** (jamais « égaux ») ; entre les deux = **non résolu**, publié
tel quel. Demi-étendue prise sur la phase de **mesure** (préreg 08-11 §4).
🚨 **`Δ_contrôle` n'est imprimé que si `n_phases > 1`** (`:1953`) : un job à une
phase ne produit **aucun R**, donc aucune règle, et rien ne le signale — le
plan de phases est une **condition de validité**.

**2.3 — Ordre de dispatch.** Les sept bras sont
`slot32, planes14, planes12x, golay70v1, fp16, awq, golay70v2` (`arms.rs:45-47`).
**Tout bras neuf s'ajoute EN DERNIER** (`:116-121`). `LLVQ_BENCH_ARMS` **saute**
des bras, n'en **déplace** jamais.

**2.4 — Plan de phases.** Chaque phase contient `fp16` (`arms.rs:154-160`) et
est un **sur-ensemble** de la précédente (`:161-173`).

| phase | bras | rôle |
|---|---|---|
| **1** | les **six** du 08-10 (sans `golay70v2`) | reproduit la phase 1 du run publié |
| **2** | les **sept** | **le contrôle** — c'est la phase 2 du 08-11 (`docs/data/README.md:20`) que la table publiée mesure, et elle fabrique `Δ_contrôle` |
| **3** | les sept **plus** les bras neufs | la table de P4 |

**2.5 — Bras neufs, tous à écrire**, dans cet ordre après `golay70v2` :

| bras | ce qu'il est | existe ? |
|---|---|---|
| `cublasf16` | le **dénominateur publiable** : f16 non handicapé, à k | ❌ |
| `mvkf16` | matvec-k f16 **maison** — **CONTRÔLE**, jamais publié seul (§4.3) | ❌ |
| `nullk` | même grille, même k, sortie écrite, **aucune lecture de poids** | ❌ |
| `planes14k`, `planes12xk`, `golay70v2k` | les trois layouts en k-colonnes, **k ∈ {1, 2, 4, 8}** | ❌ |
| `e1c14`, `e1c12` | X3 — les deux flux transposés, **k = 1 seulement** | ❌ (noyau **et** bras) |

⚠️ **`mvkf16` est un contrôle, et le dossier a payé pour l'apprendre** : le
piège `broadcast_matmul` (`llvq-llm/src/model.rs:553`) a fait publier un ×2,03
venu d'un défaut de **notre** bras dense. **Tout chiffre de débit se cite en
deux formulations** (brut, et contre un dénominateur non handicapé) — règle
actée en HISTORIQUE (`docs/HISTORIQUE.md:125-126`), **non encore reportée dans
le CLAUDE.md** : `docs/PLAN.md:140` est la case **ouverte** qui le demande.

**2.6 — La forme du bras k, figée, parce qu'elle décide de ce que K2 mesure.**
k porté par `grid_dim.y` ⇒ chaque bloc relit et redécode ⇒ **aucun
amortissement**, seulement du remplissage de grille. k porté par des
**accumulateurs en registres, grille inchangée** ⇒ poids lus et décodés **une
fois**, utilisés k fois ⇒ **c'est l'amortissement que K2 mesure**, et c'est ce
qui donne son sens à K3. **Forme retenue : accumulateurs, grille inchangée** ;
un bras k écrit autrement ne porte aucun verdict.

**2.7 — Une seule unité de traduction, des noyaux distincts.** `TILE_BLOCKS`
est injecté par `#define` (`:750`) et `arms.rs:24-27` pose que l'unité de
traduction **ne change jamais**. Les variantes k sont donc **compilées dans la
MÊME unité**, en noyaux distincts (`tv_planes_k1`, `_k2`, `_k4`, `_k8`, …),
chacun portant sa constante par un second `#define` ; **un seul sha256** couvre
tout. Un plan produisant plusieurs sha256 est une **dérogation à déclarer au
§7bis avant le job**.

**2.8 — La tuile, et le plancher qui la protège.** `shared = TILE_BLOCKS × DIM
× 4`, avec `TILE_BLOCKS = 128` et `DIM = 24` chez l'incumbent (`:97`, `:1496`,
`planes.cu:43-50`) = **12 288 o/bloc** [calculé]. Deux effets décident : **sous
32 blocs par tuile le warp est partiellement inactif** — la boucle est
`for (u32 j = jlo + lane; j < jhi; j += 32u)` (`planes.cu:52`, `planes12.cu:73`,
`golay70.cu:75`), donc à 16 blocs seize lanes sur trente-deux n'ont rien à
faire ; **à exactement 32 chaque lane traite un bloc** et le staging (copie
partagée + deux `__syncthreads`) n'est plus amorti. **Règle : `TILE_BLOCKS_K =
32` pour TOUTE la famille k, k = 1 compris**, d'où un partagé de `k × 3 072 o` :

| k | partagé/bloc [calculé] | limite driver 49 152 o/bloc | blocs/SM par le seul partagé (102 400 o/SM) |
|---|---|---|---|
| 1 | 3 072 | ✅ | 33 |
| 2 | 6 144 | ✅ | 16 |
| 4 | 12 288 | ✅ | 8 |
| 8 | 24 576 | ✅ | **4** |

[limites lues au driver, `cuda-preflight-2026-08-05.txt:29` ; reste calculé]

🚨 **Confondant qui joue CONTRE k** : ce plafond tombe de 16 à 4 entre k = 2 et
k = 8, alors que le compte de journal à k = 1 est de 6 blocs/SM. **Un K2
atteint est donc conservateur ; un K2 manqué est ambigu** entre « pas
d'amortissement » et « occupation perdue » — d'où l'issue « K2 inattribuable »
du §6, qui est la lecture décidée de ce cas.

**2.9 — Les trois séparations du sous-remplissage.** 🚨 À n = 1, quatre des cinq
familles de forme sous-remplissent la carte
(`attribution-cuda-2026-08-05.txt:86-102`, [mesuré]) :

| forme (d_out × d_in) | blocs à k=1 | LLVQ Go/s | vs FP16 |
|---|---|---|---|
| 1024 × 2560 (`k_proj`, `v_proj`) | **128** — 15 % de la capacité | **157** | **1,06× — RIEN** |
| 2560 × 4096 · 2560 × 9728 | 320 | 355 · 453 | 1,68× · 2,17× |
| 4096 × 2560 | 512 | 363 | 1,70× |
| 9728 × 2560 (`gate`, `up`) | **1216** | **469** | 1,97× |

1. **Grille constante en k** (§2.6) : le remplissage ne varie pas avec k.
2. **Partition figée PAR LISTE DE FORMES, pas par une capacité.** Le
   sous-ensemble saturé est, **pour tout k**, exactement les **72 matrices
   `gate_proj` et `up_proj`** (d_out = 9728, `planesbench.rs:129-130`). Toute
   autre partition — y compris redéfinie sur une capacité recomptée à k > 1 —
   est **post hoc et ne porte aucun verdict** ; la capacité recomptée est
   publiée comme observation.
3. **La part fixe par lancement est mesurée.** Une passe, c'est **252
   lancements** pour ~5,1 ms (`golay70-v2-sept-bras-2026-08-11.txt:18`) ; à
   grille inchangée leur nombre ne bouge pas avec k, donc tout coût fixe
   s'amortit sur k **sans être de l'amortissement de décodage**. `nullk` est
   dispatché **à chaque k dans les mêmes rounds** et borne cette part.

**2.10 — Chronométrage par forme, en rounds SÉPARÉS.** Le banc ne pose
aujourd'hui **qu'un** chronomètre : un `Instant::now()`, la boucle
`for m in &mats`, **un** `cuda.sync()`, **un** `t.elapsed()` (`:1919-1935`) — le
sous-ensemble saturé n'est extractible d'aucun chiffre produit. Décision : **les
bras k sont chronométrés par forme, par events CUDA encadrant les matrices de
chaque forme, dans des rounds séparés** — jamais un `cuda.sync()` par matrice,
qui supprimerait le recouvrement et changerait l'objet. Le précédent est écrit
dans le dépôt : les events par matrice « are not free, so these rounds are
separate and their totals are NOT the published milliseconds »
(`matvec.rs:719-721`). **Le chronomètre de passe reste inchangé** et produit la
table des sept bras. **Si ce mécanisme n'est pas écrit avant le job, K1 et K2 ne
sont rendus ni en vert ni en rouge** (§7).

**2.11 — Ordre dans un round.** **`k ∈ {1, 2, 4, 8}`, et rien d'autre** ; tout
point supplémentaire est une **dérogation à consigner au §7bis**. Dans chaque
round : les incumbents d'abord dans l'ordre d'`ARM_NAMES`, puis la famille k, k
croissant, bras dans l'ordre d'`ARM_NAMES`. Chaque couple (bras, k) est dispatché
**à chaque round**, sans quoi aucun rapport ne se forme round par round.

**2.12 — Six sites, pas un.** Tout bras neuf est ajouté **nommément** à
`arms::ARM_NAMES` (`arms.rs:45-47`), `DISPLAY` (`planesbench.rs:1705-1713`),
`ROW_NAMES` (`:1714-1722`), `arm_bytes` (`:399-409`), `verify_arm` (`:1583`) et
la boucle de dispatch (`:1920-1932`), plus la liste du gate zéro-spill
(`:808-827`). ⚠️ `arm_bytes` finit par `_ => unreachable!` (`:407`) : un site
oublié fait **paniquer le banc au rapport, après ~25 min de carte payées**.

**2.13 — A4 (fusion), et ce que la sélection ne réduit pas.** A4 se déclenche dès
que `slot32` **et** `planes14` sont sélectionnés (`:2028-2035`) ; la désactiver
casserait le contrôle. **Décision : A4 tourne, ses rounds ne sont ni lus ni
publiés en P4** ; elle vit après la table (c'est pourquoi elle y a été déplacée,
après avoir biaisé les bras à correction de ~+0,4 %, `:1445-1466`). Et le
**transcode hôte Slot32 est INCONDITIONNEL** (`:914-919`, message `:1324-1325`) :
c'est le contenu contre lequel tout bras LLVQ est prouvé. Les autres
constructions suivent l'**UNION** des phases (`:922-940`, `:965-970`), pas la
phase courante — **une phase pauvre n'économise pas une seconde de
construction**.

**2.14 — Résidence VRAM, déclarée d'avance** (aucun journal ne l'imprime) :
~17,2 Go à 6 bras [estimé, préreg 08-10 `:278-282`] ; `golay70v2` partage les
tampons de `v1` (+0) ; les flux E1c ajoutent **2,07 Go (E1c14) et 1,71 Go
(E1c12)** [mesuré, compte hôte, `rtbits-e1c-4b-2026-08-12.txt:69-70`] ; A4
~2,9 Go [estimé, `:1457`] ; les bras k réutilisent les mêmes flux (+0) et
n'ajoutent qu'un `d_y` de `k × 9 728` f32 [calculé : 311 296 o à k = 8]. Soit
**~24 Go** [calculé sur des estimés — ordre de grandeur], sous les 48 Go de la
carte. 🚨 **`cublasf16` ajoute un handle et un workspace que `Staged`
(`:272-303`) ne comptabilise pas : part INCONNUE.**

**2.15 — cuBLAS sort du dispositif de traçabilité. Quatre trous.** (1) **Hors de
l'unité NVRTC**, donc **il ne change pas le sha256** (`arms.rs:24-27`,
`:751-770`) — palliatif exigé : **imprimer la version de la bibliothèque cuBLAS
chargée**. (2) **Absent de tout rapport de registres** : **le gate zéro-spill ne
le couvre pas** (liste en dur des 9 noyaux NVRTC, `:808-827`) ; K3 ne dit rien de
lui. (3) **Résidence non comptabilisée**, et il échappe au garde qui fait échouer
le banc si `binary_version ≠ ARCH_BINARY_VERSION` (`gpu.rs:339-345`). (4) **Le
témoin FP16 n'est pas le checkpoint du modèle** : `matvec.cu:403-406` dit que `w`
porte la **reconstruction LLVQ arrondie en binary16, dans la base tournée**, « a
baseline of *cost*, not of quality » — **`cublasf16` doit lire EXACTEMENT ce
contenu**, sinon il mesure un autre objet. ⚠️ Sa dispersion sera probablement
plus large (cuBLAS **choisit ses algorithmes par lancement**, `smoke.rs:58`) ;
elle entre dans `R` comme celle de tout le monde et peut, seule, faire monter la
barre du job — c'est ce que plafonne le §4.4.

**2.16 — Traçabilité du run.** `LLVQ_KERNEL_DIR` remplace les sources et le banc
l'annonce (`:772-792`) : un journal portant « SOURCES … SURCHARGÉES » ne se
rattache qu'au sha256 imprimé, et **vérifier leur absence fait partie de la
lecture du journal**. La ligne de coût **s'écrit à la main** : **aucun code
n'écrit `docs/data/jobs.csv`** (`grep -rn "jobs.csv" ops/` ne rend qu'un
commentaire, `ops/run.py:1360` ; `docs/data/README.md:23` en attribue à tort la
source au « moniteur »). Elle sera recopiée depuis `monitor`, qui calcule sur le
temps **facturé** (`durations.running_secs × usd_h/3600`, `ops/run.py:1221-1232`),
jamais sur l'horloge murale (35,80 $ d'horloge pour un run à 11,48 $, ibid.).

## 3. V0 avant V1 — l'exactitude d'abord, sans exception

**Aucune milliseconde n'est chronométrée avant que chaque bras neuf soit
prouvé.** Dans cet ordre, et le journal doit le montrer :

1. **Chaque bras neuf porte SA PROPRE référence f64** (08-10 §7), **ligne à
   ligne**, sur les **1 105 920 lignes** des 252 matrices, seuil `1e-5` (`TOL`,
   `:102`). Pires erreurs attendues dans 2,2–3,0·10⁻⁸.
2. **Le seuil se fixe par le FORMAT, jamais par ce qui passe.** Précédent : le
   `1e-3` d'AWQ (`AWQ_TOL`, `:110`), justifié par une sortie binary16 (~2⁻¹¹ par
   construction, `:103-109`). `cublasf16` aura le sien, **posé avant de voir son
   erreur**.
3. **Égalité BIT À BIT entre bras au même contenu décodé** : `e1c14` ↔
   `planes14`, `e1c12` ↔ `planes12x`, qui **ne diffèrent que du bourrage** — ce
   que le repack hôte a prouvé sur les **150 681 600 blocs** du 4B scellé
   (`e1c-sweep-4b-2026-08-12.txt:2,17`). Le noyau CUDA doit le reproduire, pas
   l'hériter.
4. **Le point k = 1 de la famille k est un contrôle de RÉSULTAT, pas de temps.**
   `planes14k` à k = 1 doit rendre le même contenu décodé que `planes14` (f64
   ligne à ligne, et bit à bit). ⚠️ **Leurs TEMPS ne sont pas comparables** : la
   famille k tourne à `TILE_BLOCKS_K = 32`, l'incumbent à 128 (§2.8) ; l'écart
   est **publié comme coût de géométrie de tuile — observation, aucun verdict.**
   Tous les rapports en k se forment **à l'intérieur de la famille k**, jamais
   contre l'incumbent.
5. **Le gate zéro-spill doit être ÉTENDU nommément aux noyaux neufs** (liste en
   dur, `:808-827`) : un noyau absent passerait le gate **sans être testé**.
6. **Le sweep hôte E1c ne dispense de rien** : il prouve le **format**, pas le
   **noyau**.

🚨 **`preflight` ne couvre PAS les noyaux du banc** : il ne charge que
`load_sources()` et n'inspecte que `decode_probe`, `dot_probe`, `floor_probe`
(`preflight.rs:68-69`, `:115`). Le rapport de registres d'un noyau k sortira de
`planesbench`, **après le transcodage — après ~25 min de carte payées.**
Accepté d'avance : un K3 rouge coûte le prix du transcode, et ce n'est pas une
raison pour le sauter. **Tout écart enterre le bras sans banc** (règle 08-11).

## 4. Les seuils, posés avant la première ligne de code

**Tous les seuils de ce §4 sont soumis à la règle `R` du §2.2** : un écart au
seuil inférieur à `R` est publié **NON RÉSOLU**, jamais comme un vert.

### 4.1 Le balayage en k — K1, K2, K3

Repris de la passation (`passation-exec-2026-08-13.md:162-168`), figés ici. Tout
se juge sur la **médiane des rapports round par round**, sur les **72 matrices
du sous-ensemble saturé** (§2.9.2), **à l'intérieur de la famille k** (§3.4).

| | critère | atteint | sinon |
|---|---|---|---|
| **K1** | `R(golay70v2k) < 0,95 × R(planes12xk)` **AU MÊME k**, aux **trois** points **k ∈ {1, 2, 4}**, où `R` est le **rapport vs FP16** (médiane des rapports formés round par round) — soit, en temps, `T(golay70v2k) > 1,0526 × T(planes12xk)` | **la famille des décodeurs lourds se referme** — le déficit de Golay70 n'est pas un défaut d'amortissement | la famille reste ouverte : l'amortissement récupère ce que n = 1 cachait, l'axe se rejuge |
| **K2** | **`T(golay70v2k, k=8) / 8 ≤ 0,60 × T(golay70v2k, k=1)`**, c'est-à-dire **`T(k=8) ≤ 4,80 × T(k=1)`** | le modèle d'amortissement tient : le décodage se paie une fois pour k colonnes | **le modèle d'amortissement est FAUX** ; le package prefill perd son fondement chiffré |
| **K3** | **zéro-spill** (`local_bytes == 0`) avec **k accumulateurs**, à chaque k mesuré | k reste libre jusqu'à 8 | **k plafonne à 4** ; les points au-delà ne sont pas publiés |

🚨 **K1 se lit sur le RAPPORT vs FP16, pas sur le temps — et l'inversion a
failli passer.** Une version antérieure de cette table écrivait
`T(golay70v2k) < 0,95 × T(planes12xk)`, c'est-à-dire « Golay70 est **plus
rapide** que Planes12x », alors que le sens visé est « Golay70 ne récupère
**pas** son déficit ». Les deux formes concluent l'inverse l'une de l'autre sur
les données déjà en main : au run à sept bras du 2026-08-11, Planes12x rend
**5,502 ms / 2,00×** et Golay70 v2 **6,214 ms / 1,77×**
([`docs/mesures/golay70-v2-sept-bras-2026-08-11.txt`](../docs/mesures/golay70-v2-sept-bras-2026-08-11.txt)) —
donc `R(golay)/R(planes12x) = 0,885 < 0,95` (**la famille se referme**) mais
`T(golay) = 6,214 > 0,95 × 5,502 = 5,227` (**la famille resterait ouverte**).
Le §9 de ce même document lisait déjà l'orientation correcte pendant que le
tableau l'écrivait à l'envers : c'est une **contradiction interne**, et elle
aurait été tranchée après la mesure par celui qui lisait la bonne moitié.
Les deux écritures équivalentes sont désormais dans la ligne, pour qu'aucune
lecture ne subsiste.

🚨 **K2 se lit PAR COLONNE, et la forme absolue est écrite à côté.** Un noyau à
k = 8 fait 8× les FMA et 8× les stores de k = 1 : `T(k=8) > T(k=1)` est
arithmétiquement forcé, et un seuil écrit `T(k=8) ≤ 0,60 × T(k=1)` ne pourrait
**que** échouer — **un seuil dont on connaît le verdict d'avance n'est pas un
critère**. Sous la forme retenue les deux issues sont atteignables : sans
amortissement `T(k=8) ≈ 8 × T(k=1) > 4,80` ; avec amortissement complet le
surcoût se réduit aux FMA et aux stores. **K2 se juge sur la forme NETTE** —
temps diminués de la part fixe mesurée par `nullk` au même k (§2.9.3), parce que
c'est ce que le modèle prédit. **La forme brute est publiée à côté ; si les deux
rendent des verdicts opposés, K2 n'est pas rendu et les deux sont publiées.**

🚨 **K1 est RELATIF, au même k.** Un absolu recyclé de n = 1 serait un
**déplacement de poteaux** : à k = 4, `Planes12x` monte aussi. Corollaire de la
règle §4 du 08-10 : *aucun rapport n'est cité contre un jeu de bras — ni ici
contre un k — qui ne l'a pas produit.* ⚠️ **Et K1 se juge sur le même
sous-ensemble saturé que K2**, argument posé avant la mesure : le §2.9 établit
que 72 des 252 matrices rendent « 1,06×, c'est-à-dire RIEN » ; si le
sous-remplissage disqualifie l'agrégat pour K2, il le disqualifie pour K1, qui
compare deux décodeurs de coûts ALU très différents précisément sur des formes où
ni l'un ni l'autre n'est borné mémoire. **L'agrégat des 252 est publié pour
mémoire, jamais comme le chiffre de K1 ni de K2.**

⚠️ **K3 est NÉCESSAIRE, PAS SUFFISANT**, écrit d'avance depuis le 08-10
(`:247-249`) : un bras à 0 octet local peut avoir **perdu son occupation** — k
accumulateurs, c'est k × les registres — et `num_regs` **ne la borne pas**.

### 4.2 Les seuils X3 (E1c) — une ligne PAR BRAS, jugés à k = 1

Le spec attache chaque seuil à un layout **nommé** : X1 = `e1c14`, X2 = `e1c12`
(`spec-memoire-extreme-2026-08-12.md:163-165`). On garde une ligne **par bras** :
les deux sont mesurés dans le même run et `e1c12` lit 0,79 b/poids noyau de moins
qu'`e1c14`, donc « `e1c12` = 2,10×, `e1c14` = 1,95× » est plausible.

| bras | verdict | critère |
|---|---|---|
| `e1c14` | **remplace `Planes14` en production** | rapport médian ≥ celui de **`Planes14` MESURÉ DANS CE RUN**, avec **séparation** au sens de `R`, **ET** ≥ **2,05×** vs FP16 |
| `e1c14` | **admis** point de fonctionnement 70B **dense** | ≥ **1,6×** vs FP16 |
| `e1c14` | **mort** | < **1,6×** |
| `e1c12` | **remplace `Planes12x`** dans l'échelle | rapport médian ≥ celui de **`Planes12x` MESURÉ DANS CE RUN**, avec **séparation** au sens de `R`, **ET** ≥ **1,9×** vs FP16 |
| `e1c12` | **admis** point de fonctionnement 70B **dense** | ≥ **1,6×** vs FP16 |
| `e1c12` | **mort** | < **1,6×** |

🚨 **Case croisée fermée nommément : un `e1c12` ≥ 2,05× NE remplace PAS
`Planes14`.** Ce seuil vise `e1c14` ; un layout ne prend pas la place d'un autre
parce qu'il a passé le seuil d'un troisième. Et **l'échelle ne se referme côté
transposition que si les DEUX bras sont sous 1,6×**, pour la cible **dense
seulement** (§0.4). ⚠️ **Pourquoi relatif au run, contrairement au spec** : un
absolu recyclé produirait l'absurdité que le §4.1 interdit — `Planes14` a rendu
2,15× [2,15–2,16] le 08-11 et 2,14× [2,11–2,15] le 08-07, donc un `e1c14` à
2,06× « remplacerait en production » un incumbent **plus rapide que lui**. Le
2,05× du spec reste le plancher ; il ne suffit pas.

### 4.3 Le contrôle `mvkf16` — son critère, posé d'avance

| écart `cublasf16` vs `mvkf16`, au sens de §2.2 | conséquence |
|---|---|
| `cublasf16` plus rapide de **> 10 %** avec **séparation** | **notre dénominateur maison est déclaré handicapé** : tous les rapports de P4 se citent contre `cublasf16`, ceux contre `tv_f16` deviennent la formulation interne |
| écart **sous `R`** | `tv_f16` reste le dénominateur ; `cublasf16` ne sert qu'à l'attester |
| entre les deux | **non résolu** : les deux dénominateurs publiés côte à côte |

### 4.4 Le plafond de résolution — une clause qui peut annuler le job

**Si `R > 1,0 %`, aucun seuil du §4 n'est rendu** : le job se republie comme
points de courbe et toutes les questions restent entières. Nombre posé
maintenant, pas après. Il est atteignable : le run du 08-11 rend `R = 0,56 %` à
sept bras (`golay70-v2-sept-bras-2026-08-11.txt:39-40`), la phase 3 ajoute au
moins huit bras, **plus** 3,78 Go de flux E1c, **plus** la dispersion de cuBLAS
(§2.15). Cette clause ne peut que **fermer**, jamais sauver un bras d'un rouge.

### 4.5 Le bras cascade/marche CUDA — conditionné, pas présupposé

**Autorisé seulement si P1 rend ≤ 0,45 ns/bloc** sur le meilleur de ses deux
décodeurs (`proofs/preregistration-p1-2026-08-13.md:252-259`). **Ce document ne
le présuppose pas** : à la signature, P1 n'a aucune mesure et ces décodeurs
aucune ligne de code. Sans verdict P1 avant le lancement, **le bras n'est pas
dans le job**, et son absence ne s'interprète pas comme un résultat.

### 4.6 Le budget — et pourquoi le chiffre de la passation est faux

🚨 **La passation annonce « ~0,3–0,5 $ » (`:152`). C'est faux dès que
`Planes12x` ou `Golay70` entrent** : le poste dominant est le **transcodage hôte
avant le premier round**.

| job | bras | transcode hôte [mesuré] | facturé [mesuré] |
|---|---|---|---|
| `c1-planesbench` (08-06) | 3 (2 LLVQ) | **150 s** | 3 min / **0,08 $** |
| `e2-golay70-bench` (08-07) | 5 (4 LLVQ) | **1 468 s** | 25 min / **0,74 $** |
| `baseline-head` (08-10) | 5 | — | 26 min / **0,78 $** |
| `six-arm-awq` (08-10) | 6 (4 LLVQ + AWQ) | **1 481 s** | 26 min / **0,78 $** |
| `golay70-v2-sept-bras` (08-11) | 7 | — | 26 min / **0,77 $** |

[transcodes : `c1-planesbench-2026-08-06.txt:14`,
`e2-golay70-bench-2026-08-07.txt:16`, `six-arm-awq-2026-08-10.txt:59` ; coûts :
`docs/data/jobs.csv:15,19,35,36,37` ; tarif l40sx1 **1,80 $/h = 0,0300 $/min**,
table codée en dur `ops/run.py:84`, pas un relevé de facture]

**Et ce poste ne se réduit PAS en désélectionnant** (§2.13). Donc : **budget
annoncé 0,8–1,0 $** [estimé — analogie avec les jobs 5-7 bras, plus la marge du
balayage en k et des deux flux E1c ; **le transcode d'un flux k ou E1c n'est
mesuré nulle part**, ABSENT assumé, §8] ; **plafond dur** : `--timeout` **posé
explicitement**. ⚠️ Son défaut est **30 m** (`ops/run.py:1421`) — la docstring de
la même fonction affirme le contraire (`:901-902`, « mandatory and has no silent
default ») — et **le transcodage seul dure ~25 min** : le défaut tuerait le job
après avoir payé la construction. À **90 m** le pire cas est **2,70 $** [calculé :
90 × 0,0300], imprimé à chaque lancement (`:964-971`). **Le go de dépense porte
sur 0,8–1,0 $ attendus, 2,70 $ de pire cas** ; cumul rapporté après.

## 5. La prédiction, et ce qui ne la fonde pas

**Aucune fourchette n'est prédite pour les bras k : ils n'ont aucune ligne de
code.** Un compte d'instructions sur du code non écrit serait un vœu. **Aucune
pour E1c non plus** : le dossier possède un **compte de bits exact**, pas un
temps — E1c14 **4,5551** et E1c12 **3,7618** b/poids noyau [compte hôte exact,
sans dispersion, `rtbits-e1c-4b-2026-08-12.txt:69-70`] contre 4,8040 et 4,3424.
Le CLAUDE.md exige que leur colonne « vs FP16 » **reste vide jusqu'à la carte** ;
ce document ne la remplit pas.

🚨 **Ce qui NE fonde PAS les seuils :**

- **Le 0,60 de K2 n'est dérivé d'aucun modèle mesuré** — jugement d'ingénierie
  (« si le décodage s'amortit, huit colonnes doivent coûter nettement moins que
  huit fois une »). **C'est le maillon faible du document**, comme le 0,45 ns
  l'était pour P1 : il transporte une intuition dans un seuil chiffré. Posé
  d'avance pour ne pas être négocié après coup, **pas parce qu'il est solide**.
  **Le plafond de 1,0 % du §4.4** est du même bois : un seul précédent.
- **Un compte niveau source a déjà manqué sa cible sur ce noyau, par le haut** :
  Golay70 v2, **1,9–2,4× estimé**, **1,77× mesuré**, le compte ALU ÷2 ne s'étant
  pas traduit en temps ÷2. **Aucun compte de P4 ne vaut mieux**, et aucun n'est
  un profil.
- **Aucun sol de bande passante** (§0.1) : aucun seuil ne s'exprime en « % du
  pic ». **Aucun profil n'existe et aucun ne sera produit** (§0.2).
- **La partition de K1/K2 est stable par DÉCISION, pas par mesure** (§2.9.2),
  précisément parce que la capacité de 852 blocs est un compte de journal qui
  bougera avec k. Publié comme tel.

⚠️ **Si un bras k rend mieux que `0,50 × 8 × T(k=1) = 4,00 × T(k=1)` à k = 8,
chercher l'erreur avant d'en faire un titre** : grille non constante (§2.6),
activation non comptée (§1.4), part fixe non retirée (§2.9.3), bras dégradé,
contrôle de résultat k = 1 non reproduit (§3.4), sortie non observable et boucle
éliminée.

## 6. Les issues, et ce que chacune fait au dossier

| issue mesurée | conséquence, décidée d'avance |
|---|---|
| **K1 atteint** aux trois points k ∈ {1,2,4} | **la famille des décodeurs lourds se referme, cible DENSE seulement** (§0.4) ; E2 reste clos, le déficit de Golay70 est acté structurel |
| **K1 manqué** à un k | la famille **reste ouverte** ; l'axe se rejuge et le 2,00× de n = 1 **n'est pas recyclé** |
| **K1 non résolu** (écart < `R`, ou `R` > 1,0 %) | **aucun verdict** ; temps publiés en points de courbe, axe ouvert, **le 0,95 n'est pas abaissé** |
| **K2 atteint** (forme nette, sous-ensemble saturé) | le modèle d'amortissement tient ; le **package prefill** garde son fondement chiffré |
| **K2 manqué** | **le modèle d'amortissement est faux** ; le package prefill se réécrit — **pas de repêchage sur l'agrégat des 252**, qui ne porte aucun verdict |
| **K2 inattribuable** (grille non constante ; brute et nette en désaccord ; occupation visiblement tombée, §2.8) | **aucun verdict K2** ; temps publiés en points de courbe, la question reste entière |
| **K2 non rendu** (chronométrage par forme non écrit, §2.10) | **ni vert ni rouge** ; le job ne conclut pas sur l'amortissement — et le préreg le disait avant |
| **K3 manqué** | **k plafonne à 4** ; les points k = 8 ne sont pas publiés, et K2 — qui s'y lit — **n'est pas rendu** |
| **`e1c14` ≥ 2,05× ET > `Planes14` du même run, avec séparation** | `e1c14` remplace `Planes14` en production ; câblage `LLVQ_FUSED_LAYOUT` + A/B `fusedrun` (tokens gloutons identiques au dense), **go de dépense séparé** |
| **`e1c12` ≥ 1,9× ET > `Planes12x` du même run, avec séparation** | `e1c12` remplace `Planes12x` dans l'échelle ; le b/poids noyau tombe de 4,3424 à 3,7618 |
| **`e1c14` ou `e1c12` ∈ [1,6× ; son seuil de remplacement[** | **admis** point de fonctionnement 70B **dense** seulement, publié comme point de la courbe débit↔taux |
| **`e1c14` non résolu** / **`e1c12` non résolu** | **aucun verdict pour ce bras** ; point de courbe, axe ouvert, **le seuil n'est pas abaissé** |
| **les DEUX sous 1,6×** | **l'échelle se referme côté transposition, cible DENSE seulement** (§0.4) ; seule voie restante X4, E3 étant enterré sur papier (3,0444 contre 2,60) |
| **un seul des deux sous 1,6×** | **ce bras-là est mort** ; l'autre garde son verdict propre et **l'échelle ne se referme pas** |
| **cuBLAS non écrit à temps** | job **sans dénominateur publiable** : tous les rapports restent internes (contre `tv_f16`) et **aucun chiffre de P4 n'entre au papier** avant qu'un dénominateur non handicapé existe |

**K3 n'a pas d'issue « non résolue »** : `local_bytes` est un entier, pas une
mesure dispersée. **Aucune de ces issues ne rouvre les seuils de P1**, qui sont
un autre document et un autre matériel.

## 7. Ce qui invaliderait ce pré-enregistrement

- **journal sans les 252 matrices ni le nom du fichier** ⇒ run **synthétique**,
  nul et non avenu (§1.6) ;
- **une seule phase déclarée** ⇒ pas de `Δ_contrôle` (`:1953`), donc pas de `R`,
  donc **aucun seuil rendu** — et rien dans le log ne le signale ;
- **la phase de contrôle est conforme si et seulement si** (i) les **b/poids
  noyau** des sept bras retombent **AU CHIFFRE** sur ceux du 08-11 — compteurs
  hôte sans dispersion (préreg 08-10 `:284-286`) — **et** (ii) les **plages** de
  rapports vs FP16 des sept **recouvrent** celles publiées le 08-11. Une plage
  disjointe **suspend la publication** jusqu'à cause connue. ⚠️ Aucun `R`
  inter-processus n'est invoqué : la règle §2.2 est intra-processus par
  construction (`:1949`) ;
- **b/poids noyau d'`e1c14` ou `e1c12` ne retombant pas AU CHIFFRE sur la valeur
  hôte de `rtbits` (4,5551 et 3,7618)** ⇒ le flux monté n'est pas celui que le
  sweep de 150 681 600 blocs a validé : **aucun verdict**, quelle que soit la
  vérification f64 ;
- **échec d'une vérification f64 ligne à ligne** ⇒ le bras n'existe pas ;
- **`e1c14`/`e1c12` ne rendant pas un contenu décodé BIT À BIT identique** à
  `planes14`/`planes12x` ⇒ « même contenu, moins le bourrage » est faux **côté
  noyau**, et le sweep hôte ne le rattrape pas ;
- **bras k à k = 1 ne rendant pas le même contenu décodé que son incumbent**
  (§3.4) ⇒ aucun verdict ;
- **géométrie de grille non constante en k** (§2.6) ⇒ le bras mesure du
  remplissage, K2 n'est pas rendu ;
- **`TILE_BLOCKS_K ≠ 32` sur toute la famille k** (§2.8) ⇒ les points ne sont
  pas comparables entre eux, aucun rapport en k n'est rendu ;
- **chronométrage par forme non écrit** (§2.10) ⇒ **K1 et K2 ni verts ni
  rouges** ;
- **`cublasf16` ne lisant pas exactement le contenu de `tv_f16`** (§2.15.4) ⇒ ce
  n'est pas un dénominateur ;
- **noyau neuf absent de l'un des six sites** (§2.12) ou de la liste du gate
  zéro-spill ⇒ le gate passe **sans l'avoir testé**, un vert qui ne prouve rien ;
- **journal portant « SOURCES … SURCHARGÉES »** (§2.16) ⇒ le run ne se rattache
  qu'à son sha256.

## 7bis. Écarts au protocole — journal, tenu à chaud

*(Chaque entorse s'écrit ici le jour où elle est commise, avec sa raison et son
coût — règle du 08-10.)* **Vide à la signature** : aucun code écrit, aucun job
lancé.

## 8. Ce qui est connu à la signature — divulgation datée

**Généalogie des seuils.** K1/K2/K3 datent de la passation du **2026-08-13**
(`:162-168`) ; **K2 est reformulé PAR COLONNE ici, le 2026-08-14**, sur décision
d'opérateur, la forme héritée étant arithmétiquement impossible à passer (§4.1).
Les seuils X3 (1,6 / 1,9 / 2,05×) datent du spec du **2026-08-12**, posés avant
que le noyau existe ; ils sont ici **relativisés au run** et **dédoublés par
bras** — donc **plus durs**, jamais plus faciles.

**Mesures connues, toutes à k = 1, run 7 bras du 2026-08-11**
(`golay70-v2-sept-bras-2026-08-11.txt:16-22`, `docs/data/echelle-formats.csv`) :

| bras | b/poids noyau | méd ms | Go/s (sur le **min**) | vs FP16 (médiane des rapports) |
|---|---|---|---|---|
| FP16 (`tv_f16`) | 16,000 | 11,008 | 661 | 1,00× |
| Slot32 | 5,510 | 5,834 | 429 | 1,89× [1,88–1,89] |
| Planes14 | 4,804 | 5,116 | 427 | 2,15× [2,15–2,16] |
| Planes12x | 4,342 | 5,502 | 360 | 2,00× [2,00–2,01] |
| Golay70 v1 | 3,589 | 8,205 | 199 | 1,34× [1,34–1,34] |
| Golay70 v2 | 3,589 | 6,214 | 263 | 1,77× [1,76–1,78] |
| AWQ w4g128 | 4,179 | 3,263 | 583 | 3,37× [3,36–3,38] |

Résolution : **Δ_contrôle = 0,24 %, `R` = 0,56 %** → indiscernable (`:39-40`).
⚠️ **Ce run est un deux-phases (6 bras puis 7) et sa table publiée est sa PHASE
2** (`docs/data/README.md:20`) — c'est ce que le §2.4 reproduit, plutôt qu'une
phase 1 à sept bras, qui ne serait pas le même objet.

🚨 **Divulgation qui touche K1.** Sur l'**agrégat des 252 matrices**,
`T(Planes12x) / T(Golay70 v2) = 5,502 / 6,214 = 0,885` [calculé sur les médianes
ci-dessus] — **déjà sous le 0,95 de K1**. ⚠️ **Mais K1 se juge sur le
sous-ensemble saturé** (§4.1), et **aucun journal ne donne de temps par forme
pour Golay70 : ABSENT.** Donc **aucun des trois points de K1 n'est connu à la
signature** : l'agrégat est le seul chiffre en main, il pointe vers la
fermeture, et il n'est pas la quantité jugée.

**Autres divulgations :**

- **Aucune milliseconde E1c n'existe, dans aucune comptabilité, sur aucun
  matériel.** Existe : le sweep intégral de **150 681 600 blocs** prouvant
  l'exactitude du **repack hôte** (`e1c-sweep-4b-2026-08-12.txt:2,17`, 459,16 s
  sur M3 Max), et les b/poids ci-dessous. **Aucun noyau CUDA.**
- ⚠️ **Le b/poids payload d'E1c12 existe sous deux valeurs dans le dépôt.**
  **3,6196** = flux principal **+ exceptions** (`rtbits-e1c-4b-2026-08-12.txt:58-60` :
  3,4167 de stream + 0,2029 d'exceptions, les **mêmes 5 096 688** enregistrements
  que `Planes12x`) ; la doc de module `llvq-artifact/src/e1c.rs:18` cite
  **3,4167**, le flux principal **seul**. Les deux sont justes. Ce document
  emploie 3,6196 (payload) et **3,7618** (noyau, `:70`). Idem E1c14 : 4,4167
  payload (`:57`, `e1c.rs:16`), **4,5551** noyau (`:69`).
- **Aucune milliseconde k-colonnes n'existe** et aucun noyau maison ne prend k.
- **Le sous-remplissage de grille à n = 1 est connu depuis le 2026-08-05** et
  chiffré (§2.9). Écrit ici **avant** la mesure, parce qu'il est l'explication la
  plus commode d'un k-balayage flatteur.
- **Le coût réel des jobs `planesbench` est connu** (§4.6) : il périme le
  « ~0,3–0,5 $ » de la passation, et la correction précède le go de dépense.
- **ABSENT — le transcodage d'un flux k ou E1c n'est mesuré nulle part.** Le
  repack E1c hôte a pris 459 s sur M3 Max, mais c'est un **sweep de
  vérification**, pas un transcode de banc, et le matériel diffère. Sans
  inventer : le repack E1c **ne refait aucune recherche réseau**
  (`llvq-artifact/src/e1c.rs:21-26`), contrairement au transcode `Planes12x`
  (×4,8 sur le Mac : 404 s contre 84 s). **« Donc il sera moins cher » n'est pas
  mesuré et ne s'écrit pas comme un fait.**
- **ABSENT — aucune borne sur la résidence VRAM réelle**, et la part de
  `cublasf16` est **inconnue** (§2.14). Les ~15,3 / 17,2 / 18,4 Go du dossier
  sont de la **prose estimée** ; aucun journal ne les imprime.
- **ABSENT — la tolérance *capacity-first* n'est chiffrée nulle part** : trois
  occurrences (`etude-moe-memoire-extreme-2026-08-12.md:180`,
  `passation-exec-2026-08-13.md:106`, `preregistration-p1-2026-08-13.md:402`),
  aucune avec un nombre. D'où §0.4 : **aucune fermeture de ce document ne vaut
  pour le MoE.**
- **P1 n'a rendu aucun verdict**, donc le bras cascade/marche CUDA n'est pas
  autorisé et n'est pas budgété (§4.5).

## 9. Ce qui reste FAIBLE, et ce qui reste OUVERT

1. **Le 0,60 de K2** — jugement d'ingénierie sans modèle mesuré (§5).
2. **Le plafond `R > 1,0 %`** (§4.4) — un seul précédent, aucune théorie.
3. **Le confondant d'occupation à k = 8** (§2.8) : le partagé passe de 3 072 à
   24 576 o/bloc et la seule instrumentation est `num_regs`, qui ne borne pas
   l'occupation. **K2 vert est conservateur, K2 rouge est ambigu.**
4. **Le chronométrage par forme n'existe pas encore** : sans lui K1 et K2
   tombent. L'issue est décidée (§6), mais c'est la dépendance la plus lourde.
5. **OUVERT — le volume de code neuf** (cuBLAS, `mvkf16`, `nullk`, 3 layouts ×
   4 points de k, 2 flux E1c). Toute réduction est une **dérogation à consigner
   au §7bis avant le job**, et un critère qui lit un point manquant **n'est pas
   rendu**.
6. **OUVERT — le MoE.** Tant que la tolérance *capacity-first* n'est chiffrée
   nulle part, aucune fermeture d'ici ne s'y applique (§0.4, §8).
