# Note de design — A3, variantes d'occupation (préreg `proofs/preregistration-a2-a3-geometrie-2026-08-31.md`, sha256 802006c5…)

Statut : note de conception ; **code écrit le 2026-09-01** (`llvq-cuda/kernels/planes_occ.cu`, `llvq_cuda::occ`, section Fusion de `planesbench`, sélecteur `LLVQ_SEG_ARMS`), écarts déclarés dans le fichier d'ÉCARTS ci-dessous, aucun job lancé. Tout chiffre porte son étiquette *mesuré* / *calculé* / *estimé* et sa provenance. Écarts au préreg tamponné → `proofs/preregistration-a2-a3-geometrie-2026-08-31-ECARTS.md`, jamais d'édition du préreg.

## 1. La géométrie actuelle, au chiffre

**Le contrat de grille est un seul, partout.** `row_grid` (`llvq-cuda/src/gpu.rs:771-778`, dupliqué `llvq-cuda/src/bin/planesbench.rs:522-528`) : **un warp par ligne de sortie, 256 threads = 8 lignes par bloc CUDA**, grille exacte `d_out·32/threads`, aucun garde de bornes (un `return` avant `__syncthreads()` bloque — le host asserte `d_out % 8 == 0`). Servi et banc identiques : `THREADS = 256` (`planesbench.rs:99`, `llvq-llm/src/fused_cuda.rs:51`), `TILE_BLOCKS = 128` (`llvq-cuda/src/lib.rs:121`) soit 3 072 colonnes stagées = **12 288 o de partagée dynamique par bloc**.

**Formes servies du 4B, géométrie fusée v1** (144 lancements matvec/token, `nullkbench.rs:73-80`) :

| site | d_out×d_in | nblocks (queue) | tuiles | CTAs/lancement | remplissage de grille |
|---|---|---|---|---|---|
| qkv | 6144×2560 | 106 (16) | 1 | 768 | 90,1 % de 852 (*calculé*) |
| o_proj | 2560×4096 | 170 (16) | 2 | **320** | **37,6 %** |
| gate_up | 19456×2560 | 106 (16) | 1 | 2432 | 3 vagues, 95,1 % moyen |
| down_proj | 2560×9728 | 405 (8) | 4 | **320** | **37,6 %** |

**Le plafond par SM est atteint — la perte n'est pas là.** L40S : 142 SM, 1 536 threads/SM, 65 536 registres/SM, 102 400 o de partagée/SM (`docs/archive/passation-2026-08-05.md:95`). `tv_planes` : 40 registres, 0 o local (*mesuré*, `docs/mesures/f2-p3-qtip-banc-2026-08-21.txt:130`). Résidence : threads 1536/256 = 6 · registres 65 536/(40·256) = 6,4 → 6 · partagée 102 400/12 288 = 8,3 → non limitante. **6 CTAs/SM × 142 = 852 emplacements** — le « 852 » des commentaires de `matvec.cu` et `planes_seg.cu:20-22` (*calculé*, cohérent).

**Où ça se perd, dans l'ordre :**

1. **Grilles courtes.** À 252 lancements c'étaient k/v : 128 CTAs sur 852 = 15 %, 157 Go/s contre 469 (*mesuré*, `docs/mesures/attribution-cuda-2026-08-05.txt:98-99`). La fusion a capturé ce poste-là. **Le résidu de la géométrie v1 est o/down : 72 des 144 lancements tournent à 37,6 % de remplissage, et ils portent 35,2 % des octets** (o 10,4 % + down 24,8 %, *calculé* sur 14 o/bloc × formes).
2. **Le plancher résiduel mesuré.** A1 : nullk 144 lancements = **1,794 ms [1,793–1,796]** sans lire un octet de poids (*mesuré*, `docs/mesures/a1-nullk-252-144-2026-08-31.txt`). Dont ~0,541 ms de par-lancement (144 × 3,76 µs, *calculé*, linéarité déclarée au préreg §3 — c'est le pool d'A2) et **~1,25 ms de staging + barrières + réduction + sous-remplissage** (*calculé* par différence) — le pool d'A3.
3. **Gaspillage intra-warp de la boucle de blocs.** `for (j = jlo+lane; j < jhi; j += 32)` : à nblocks = 106, ⌈106/32⌉ = 4 itérations × 32 lanes = 128 créneaux pour 106 blocs → **17,2 % de lane-itérations vides** sur qkv/gate_up ; 11,5 % sur o (170) ; 2,6 % sur down (405) (*calculé*).
4. **Deux barrières par tuile** (`matvec.cu:124-130`), payées même quand la tuile est unique et le travail court ; et chaque CTA re-stage la même activation (10,2 Kio × 768 CTAs sur qkv — relecture L2, coût d'instructions).
5. **Les bulles sont dans le span device** : écart hôte−device 0,1–0,2 %, 4–8 µs par round entier (*mesuré*, `docs/mesures/f3-events-2026-08-19.txt:31-34`) — ce qu'A3 attaque est l'inter-noyau device (rampe/drainage de vague entre lancements du même stream), pas la soumission hôte. Et F1 borne le par-noyau : notre témoin est à ≤ 1,05× de cuBLAS (r = 1,024/1,015, *mesuré*, `docs/mesures/f1-cublasf16-2026-08-18.txt:13-15`).

**La preuve qu'une autre géométrie traverse ce plancher existe** : QTIP, `<<<128, 1024, 64 Kio>>>` figé (`planesbench.rs:843-869`), finit les 252 projections en **2,246 ms en lisant 0,91 Go, sous les 2,306 ms de notre nullk qui n'en lit aucun** (*mesuré*, F2 §2, séparation 2,7 % contre résolution 2R = 0,72 %). Sa géométrie fait les deux choses qu'A3 propose : grille fixe découplée de d_out (128 CTAs quelle que soit la forme, chaque bloc boucle sur ses paquets de lignes — `qtip_host.rs`, test :955-989 : `m_per_block = tiles_m.div_ceil(2·128)`) et **split-K intra-bloc** (32 warps se partagent les tuiles K, réduction dans le bloc).

## 2. Les bras, chacun avec son mécanisme

### Bras (a) — `tv_planes_mr` : multi-lignes par warp

**Mécanisme** : chaque warp porte R lignes (R = 2 puis 4) avec R accumulateurs ; la boucle de blocs lit R flux `planes_fields` par itération. Gains attendus : ILP sur les FMA, amortissement du staging (une tuile stagée sert 8R lignes), et ⌈nblocks/32⌉ inchangé donc gaspillage intra-warp dilué par R. Coût structurel : **la grille est divisée par R** — o/down passent de 320 à 160 CTAs (18,8 % de remplissage à R = 2), gate_up de 3 vagues à 1,43 (dernière vague 43 %).

**Changement exact** : nouveau fichier `llvq-cuda/kernels/planes_mr.cu` (variante de `planes.cu`, R injecté par `#define` comme `TILE_BLOCKS` l'est — même garde `#ifndef`, même contrat d'ordre) ; lanceur `row_grid_mr` avec `grid = d_out·32/(threads·R)` et assertion `d_out % (8R) == 0` (vrai des quatre formes fusées pour R ≤ 4, *calculé* : 6144, 2560, 19456, 2560 tous multiples de 32) ; un bras dans la section seg du banc (§3).

**Prédiction** : **−5 % à +8 %** (*estimé*) — les deux effets tirent en sens contraires et leur somme n'est pas signée sur papier ; c'est exactement pourquoi c'est le bras à ~1 jour de dev qui se mesure au lieu de se discuter. **Le plancher 1,794 ms ne le borne qu'approximativement** : même coquille warp-par-ligne mais autre grille — un jumeau `nullk_mr` (20 lignes, mêmes tampons) donnerait son plancher propre pour ~0,01 $.

### Bras (b) — `tv_planes_pers` : matvec persistant (grille-résidente)

**Mécanisme** : la grille est dimensionnée à la carte (852 CTAs = 6×142), pas au travail ; chaque CTA boucle `for (w = blockIdx.x; w < n_groupes_de_lignes; w += gridDim.x)`. Deux formes, à séparer dès le design :

- **(b1) persistant par site** — 144 lancements conservés, chaque lancement remplit toute la carte quelle que soit d_out. Supprime la quantification de vague (gate_up : plus de dernière vague) et la rampe/drainage par lancement ; sur o/down chaque CTA n'a pas plus de travail parallèle qu'avant (320 groupes pour 852 CTAs → 532 CTAs vides) — **b1 seul ne répare pas le sous-remplissage des petites formes ; il lui faut (c) ou R < 1 ligne/warp**. Portable tel quel dans `fusedrun` (un lancement reste un lancement).
- **(b2) persistant global, bras de banc seulement** — les 144 sites d'un round dans UN lancement (le banc partage un seul x et n'a aucune dépendance entre matvecs ; la liste de travail (site, groupe-de-lignes) est un tampon device statique). C'est **le concurrent direct d'A2 sans graph** : il encaisse le pool par-lancement (~0,54 ms à 144, *calculé* d'A1) EN PLUS du pool d'occupation. Garde-fou d'honnêteté, à écrire dans le journal d'avance : **b2 ne se porte pas tel quel** — dans le chemin servi les 4 sites d'une couche sont séparés par des noyaux dépendants (rot, attention, normes : `d1-fusion-servie-2026-08-24.txt`, 144 rot + 144 matvec/token) ; son chiffre borne ce qu'A2+A3 réunis peuvent viser et éclaire le kill de phase, il ne mérite pas un port à lui seul.

**Changement exact** : `kernels/planes_pers.cu` (boucle de travail au-dessus de `planes_dot` inchangé ; pour les formes à 1 tuile, stager x une fois par CTA puis boucler les lignes — pour o/down, restager par tuile comme aujourd'hui) ; lanceur à grille fixe `(6·sm_count, 1, 1)` ; pour b2, un tampon `work[]` u32 (site, offset-lignes) et les pointeurs de flux par site passés en tableaux.

**Prédiction** : b1 **+5 à +12 %** au banc (*estimé* : il récupère la rampe et les vagues, pas le sous-remplissage) ; b2 **+15 à +25 %** (*estimé* : + ~0,54 ms de par-lancement sur 4,504 ms ≈ 12 points, *calculé* d'A1, hypothèse de linéarité déclarée). **Le plancher 1,794 ms ne borne ni l'un ni l'autre** — il est le plancher de la géométrie qui lance 144 grilles taillées à d_out ; b change précisément cela (le précédent QTIP est la démonstration qu'on passe dessous, F2 §2).

### Bras (c) — `tv_planes_qg` : grille fixe + split-K intra-bloc (la leçon QTIP)

**Mécanisme** : un bloc de 256 threads possède **g lignes et TOUT K** ; ses 8 warps se partagent K en tranches fixes (warp w prend les blocs `w, w+8, …`), chaque warp accumule sa tranche, puis **réduction en mémoire partagée en ordre fixe** — déterministe, aucun `atomicAdd` inter-CTA, donc la contrainte argumentée dans `kernels/planes_seg.cu` (déterminisme, y écrit une fois) est respectée *au niveau CTA*. La grille devient `d_out·32/(threads·g)·8/… ` — dimensionnée pour que o/down remplissent la carte : avec g = 1 ligne par bloc de 256 (8 warps en split-K sur une ligne), o = 2560 CTAs → 3 vagues pleines au lieu d'une vague à 37,6 %. C'est le mécanisme qui crée du parallélisme là où d_out est petit — ce que ni (a) ni (b1) ne font. `docs/revue-taco-2026-08-22.md:279` nomme cette famille (« split-K, multi-lignes par warp ») et cadre l'attente : elle déplace le vs FP16 de ~2,15 vers ~2,5, elle ne change aucun verdict de format.

**Réserve de vérification, à poser d'avance** : le split-K **réassocie** l'accumulation — pas de comparaison bit-à-bit avec `tv_planes` possible. La preuve est celle de tous les bras : la référence f64 ligne à ligne au seuil 1e-5 (le protocole existant du banc, `planesbench.rs`, section vérification — précédent : QTIP y est tenu au même seuil).

**Changement exact** : `kernels/planes_qg.cu` (nouvelle boucle externe, `planes_dot` réutilisé tel quel ; +32·g flottants de partagée pour la réduction) ; lanceur dédié ; ne toucher **ni** `planes.cu` **ni** `planes_seg.cu` (la règle de `planes_seg.cu:1-13` : l'unité servie ne bouge pas pour un bras de banc).

**Prédiction** : **+10 à +20 %** (*estimé*). Raisonnement : o/down = 35,2 % des octets à 37,6 % de remplissage ; s'ils rejoignent le régime bien rempli (le précédent 157 → 469 Go/s du 08-05 à 15 % de remplissage donne l'ordre de grandeur du levier, pas sa valeur à 38 %), le total gagne 0,5–0,9 ms sur 4,504. C'est le bras le plus probable au-dessus du gate, et le plus cher en dev.

**Borne commune aux trois bras, et elle est étroite** : les octets. 2,182 Go lus (*mesuré*, F2) au meilleur débit soutenu du même banc — 676 Go/s, cuBLAS (*mesuré*, F1 :70) — font **≥ 3,23 ms**. Le pool disputable total au-dessus de 4,504 ms est donc **~1,28 ms ≈ 28 %** (*calculé*) ; le gate de 10 % en consomme un tiers.

## 3. Le protocole de banc

**Recommandation : étendre la section Fusion (A4) de `planesbench`, pas un binaire dédié.** Trois raisons :

1. Le gate gelé exige « `planes14` en géométrie FUSÉE, formes servies, même processus, protocole planesbench » (préreg §5) — la section seg (`planesbench.rs:2740-3064`) tient déjà la référence `planes14_seg` chronométrée dans les mêmes rounds, les flux transcodés du vrai modèle, la vérification f64 et la comparaison bit-à-bit seg/non-seg.
2. Le précédent `nullkbench` ne s'applique pas : son en-tête justifie le binaire dédié par « ce bras *mesure*, il ne candidate pas ». Les bras d'A3 **candidatent** — ils appartiennent au banc qui possède déjà leurs tampons. Un binaire dédié repaierait le transcodage du modèle (~28 min facturées sur F2) pour rien.
3. La contrainte d'`arms.rs` est ainsi contournée proprement : le registre principal (`ARM_NAMES`/`HAS_KERNEL`/`FETCHED_AT_RUNTIME`/`DISPLAY_NAMES`, toutes des `[_; N_ARMS]` appariées, N_ARMS = 17, `arms.rs:47-238`, longueurs épinglées par tests côté Mac) ne bouge pas — ces bras n'existent que sur formes fusées et n'ont rien à faire dans la table principale. La section seg a son propre tableau `tf: [Vec; 4]` (`planesbench.rs:2908`) : passer à `[Vec; 4+N]` en **appendant après les quatre historiques** respecte la règle du dépôt (jamais réordonner le dispatch d'un bras publié). Ajouter un sélecteur `LLVQ_SEG_ARMS` sur le modèle d'`arms.rs` : défaut = les quatre historiques, nom inconnu = erreur franche, jamais de repli silencieux.

**Protocole inchangé** : 7 rounds dont 2 jetés, bras entrelacés à chaque round dans l'ordre d'enregistrement, deltas et rapports formés round par round, médiane + plage, règle de résolution 2R (F2 §4 : tout écart sous 2R = non résolu). Ajouts recommandés, déclarés d'avance : (i) imprimer `gain = (t_seg − t_bras)/t_seg` round par round — la convention imprimée actuelle de la section (`planesbench.rs:3019-3024`) divise par le temps *séparé*, qui n'est pas le dénominateur du gate ; (ii) un jumeau nullk par géométrie nouvelle (plancher propre, ~0 $) ; (iii) `LLVQ_TIME_EVENTS=1` pour l'attribution par site, hors protocole publié.

**Les trois pièges d'exécution de la semaine, dans l'ordre où ils mordent :**

1. **Les DEUX listes du Dockerfile** (commit `c6642e4`) : `ops/Dockerfile.cuda:71` (build `--bin …`) ET `:116-121` (COPY). La voie planesbench n'exige **aucun** changement Dockerfile — le binaire y est déjà ; c'est un argument de plus contre le binaire dédié.
2. **L'unité NVRTC est concaténée par l'hôte, `llvq_slot.cuh` en tête** (il possède le typedef `u32` et la garde qui neutralise les `#include` — NVRTC n'a pas de système de fichiers, `nullkbench.rs:158-163`). Contrat d'ordre : `llvq_slot.cuh, matvec.cu, llvq_planes.cuh, planes.cu, planes_seg.cu`, puis les nouveaux fichiers EN QUEUE (`planes_seg.cu`, bloc final). Chaque nouveau `.cu` : garde `#ifndef`, entrée dans `bin/cuhcheck` (`cuhcheck.rs:76-88`) et compilation hôte clang++ façon `tests/host_planes.cpp` avant tout job. L'unité change pour TOUS les bras → le sha256 imprimé bouge et la référence 4,504 ne se reporte pas : **le job re-chronomètre `planes14_seg` intra-job, c'est ce chiffre-là qui est le dénominateur** (précédent R5 de D1).
3. **Le type-check croisé depuis le Mac** : `CUDARC_CUDA_VERSION=12040 cargo clippy -p llvq-cuda --target x86_64-unknown-linux-gnu --all-targets` (`docs/archive/passation-2026-08-05.md:103-105`). `planesbench.rs` est entièrement `cfg(linux)` — c'est l'incident `DISPLAY_NAMES` (`arms.rs:210-218`) : une image incompilable un jour entier parce que le Mac ne type-vérifiait jamais ce fichier.

## 4. Les critères gelés, appliqués

**Étage banc — « battre `planes14` FUSÉ de ≥ 10 % » chiffré.**

- **L40S, le chiffre à battre : `planes14_seg` = 4,504 ms** (144 lancements, médiane, 7 rounds dont 2 jetés — *mesuré*, `docs/mesures/f2-p3-qtip-banc-2026-08-21.txt:273`, job du 2026-08-21 ; même valeur citée par D1 :93). Gate à gain ≥ 10 % de ce temps → **≤ 4,054 ms** (*calculé*, 0,9 × 4,504). Réserve obligatoire : 4,504 est d'un autre processus et d'une autre unité de traduction — il **cadre** le seuil, il ne le **porte** pas ; le seuil s'applique au `planes14_seg` re-mesuré intra-job. NB : le préreg ne fixe pas la formule du « ≥ 10 % » (Δ/t_ref donne ≤ 4,054 ; un ratio ≥ 1,10 donnerait ≤ 4,095) — fixer la lecture conservatrice (Δ/t_ref) dans le fichier d'ÉCARTS **avant** le job.
- **A100, transfert (annexe, pas le gate)** : `planes14_seg` = 7,691 ms (*mesuré*, `docs/mesures/a4-a100-2026-08-31-brut/a4-a100-planesbench.txt:66`) → ≤ 6,922 ms (*calculé*). A4 a montré la géométrie invariante entre cartes (r = 0,8198 contre 0,8158, temps étirés ×1,809 ≈ l'horloge 1,787 — *mesuré*, `docs/mesures/a4-a100-2026-08-31.txt`) : un bras vert sur L40S devrait transférer, et le vérifier coûte 0,25 $.
- Le verdict au banc se rend en médiane + plage, séparé d'au moins 2R.

**Étage bout-en-bout — ≥ 8 % adopté, < 3 % clos** (préreg §5, mêmes seuils qu'A2), intra-job sur la config servie v1, tokens gloutons identiques exigés. Référence : **100,6 tok/s [99,9–100,7]** au 4B (*mesuré*, D1, gelée v1 par `vague2-fusion-8b-14b-2026-08-31.txt`) → adoption ≈ ≥ 108,6 tok/s (*calculé*, indicatif — l'A/B intra-job décide).

**L'arithmétique qui relie les deux étages, et elle est défavorable** : les 144 matvecs pèsent ~4,5 ms des 9,94 ms/token de v1 ≈ 45 % (*estimé* ; ancrage : l'économie banc de la fusion, 0,594 ms, s'est retrouvée à 0,60 ms/token dans D1 — 94,9 → 100,6 tok/s — le banc f32 se transporte ~1:1 en ms). Donc un bras **au gate exact** (10 % banc = 0,45 ms) rend **~+4,5 % bout-en-bout : entre 3 et 8 — point de courbe, non adopté**. L'adoption à ≥ 8 % exige ~0,80 ms, soit **~18 % au banc** (*calculé* sous ces hypothèses), ou la combinaison avec A2 — que le kill de phase (§6 du préreg : **A1+A2+A3 < 8 % cumulés bout-en-bout → axe géométrie sous candle clos**, cumul intra-job, jamais additif entre jobs) mesure de toute façon. Écrire cette arithmétique dans le journal du job d'avance : elle est ce qui empêchera de lire un bras à 12 % banc comme une adoption acquise.

**Ce que le plancher 1,794 ms borne, bras par bras** : (a) approximativement (même coquille, autre grille — jumeau nullk recommandé) ; (b1) partiellement (il garde les 144 lancements, ~0,54 ms de par-lancement restent le pool d'A2) ; (b2) et (c) **pas du tout** — c'est le plancher de *notre* géométrie, et F2 a mesuré un noyau réel dessous. La seule borne universelle est celle des octets : ≥ 3,23 ms, gain maximal ~28 % (*calculé*, §2).

## 5. Coût, honnête, étape par étape

| étape | dev | carte | notes |
|---|---|---|---|
| (a) `planes_mr` + jumeau nullk | 1–1,5 j | — | kernel trivial, cuhcheck + clang++ + type-check croisé : 0 $ |
| (b) `planes_pers` b1+b2 | 2–3 j | — | liste de travail device, deux lanceurs |
| (c) `planes_qg` | 2–4 j | — | réduction partagée, le plus de surface de test |
| extension section seg + `LLVQ_SEG_ARMS` | 0,5–1 j | — | 4 → 4+N bras, appendés ; impression du gain au bon dénominateur |
| **job banc L40S** (sélection `fp16,slot32,planes14`, section seg + bras A3) | — | **~0,5–0,7 $** (*estimé* : F2 à 10 bras = 0,89 $ dont 1 688 s de transcodage 5 layouts ; 2 layouts ≈ 15–20 min à ~0,03 $/min) | dans le devis préreg §5 (~0,5 $) |
| itération éventuelle (retailler g/R) | 0,5 j | ~0,5 $ | plafond de phase 4 $ (préreg §7), A2 ~0,25 $ dessus aussi |
| annexe A100 (transfert) | — | ~0,25 $ | le banc A4 a coûté 0,25 $/308 s |
| **port si gate passé** (le seul bras gagnant) | 2–4 j | ~0,25 $ | jumeau `_h` f16 dans l'unité de `fused_cuda.rs` (précédent `tv_planes_seg_h`, D1 R5), drapeau env refusant les noms inconnus, garde façon `check_fuse` pour qu'un seul mécanisme bouge par A/B ; fusedrun A/B ≈ D1 (488 s, 0,24 $) |

Total nominal : **~5–9 j de dev, ~1,0–1,7 $ de carte** avant port ; le port n'est dû qu'à un bras ≥ 10 % au banc, et l'adoption qu'à ≥ 8 % bout-en-bout. L'ordre arbitré reste **A2 d'abord** (préreg §3) — cette note prépare A3 pour qu'il parte sans latence quand A2 a rendu son chiffre, et b2 est le bras qui dira si les deux pools d'A1 (0,54 ms de par-lancement, ~1,25 ms résiduels) se convertissent vraiment. Aucun job ne se lance sans go explicite de l'opérateur, coût annoncé avant, cumul après.
